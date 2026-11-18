import time
import unittest
from unittest.mock import patch

import flask

from alerta.app import create_app, db


class ReplicaGateTestCase(unittest.TestCase):
    """
    Shared routing gate, with the feature OFF (the default on both backends).

    Guarantees that nothing changes until a replica is configured: ``get_db(readonly=True)``
    must return the primary handle and ``g.db_ro`` must never be created.
    """

    def setUp(self):
        self.app = create_app({'TESTING': True})

    def tearDown(self):
        db.destroy()

    def test_feature_off_uses_primary(self):
        with self.app.test_request_context('/alerts', method='GET'):
            self.assertFalse(db._use_replica())
            self.assertIs(db.get_db(readonly=True), db.get_db())
            self.assertNotIn('db_ro', flask.g)

    def test_gate_false_outside_request_context(self):
        with self.app.app_context():
            self.assertFalse(db._is_read_request())

    def test_gate_true_for_get_on_api_blueprint(self):
        with self.app.test_request_context('/alerts', method='GET'):
            self.assertEqual(flask.request.blueprint, 'api')
            self.assertTrue(db._is_read_request())

    def test_gate_true_for_head_on_api_blueprint(self):
        with self.app.test_request_context('/alerts', method='HEAD'):
            self.assertTrue(db._is_read_request())

    def test_gate_false_for_mutating_methods(self):
        for path, method in [
            ('/alert', 'POST'),
            ('/alert/abc/action', 'PUT'),
            ('/alert/abc', 'DELETE'),
        ]:
            with self.subTest(path=path, method=method):
                with self.app.test_request_context(path, method=method):
                    self.assertEqual(flask.request.blueprint, 'api')
                    self.assertFalse(db._is_read_request())

    def test_gate_false_for_get_on_webhooks_blueprint(self):
        # webhooks/custom.py runs the full process_alert (dedup/correlate) under GET
        with self.app.test_request_context('/webhooks/custom', method='GET'):
            self.assertEqual(flask.request.blueprint, 'webhooks')
            self.assertFalse(db._is_read_request())

    def test_gate_false_for_get_on_management_blueprint(self):
        # the management blueprint is registered as 'mgmt'; housekeeping/switchboard mutate under GET
        for path in ('/management/status', '/management/housekeeping'):
            with self.subTest(path=path):
                with self.app.test_request_context(path, method='GET'):
                    self.assertEqual(flask.request.blueprint, 'mgmt')
                    self.assertFalse(db._is_read_request())


class PostgresReplicaTestCase(unittest.TestCase):
    """
    Postgres routing, with DATABASE_URL_RO pointing at the SAME database as the primary.

    That is enough to exercise the routing (a distinct DSN opens a distinct connection);
    the actual offload to a standby is verified at runtime, not in CI.
    """

    def setUp(self):
        self.app = create_app({'TESTING': True})
        if not self.app.config['DATABASE_URL'].startswith('postgres'):
            self.skipTest('Postgres backend only')
        # same DB, different DSN spelling, so uri_ro != uri and a second connection is opened
        uri = self.app.config['DATABASE_URL']
        self.app = create_app({'TESTING': True, 'DATABASE_URL_RO': uri + '?application_name=alerta_ro'})

    def tearDown(self):
        db.destroy()

    def test_get_opens_read_only_connection(self):
        with self.app.test_request_context('/alerts', method='GET'):
            self.assertTrue(db._use_replica())
            conn = db.get_db(readonly=True)
            self.assertIn('db_ro', flask.g)
            self.assertIs(conn, flask.g.db_ro)
            self.assertIsNot(conn, db.get_db())

    def test_post_never_opens_read_only_connection(self):
        with self.app.test_request_context('/alert', method='POST'):
            self.assertFalse(db._use_replica())
            self.assertIs(db.get_db(readonly=True), db.get_db())
            self.assertNotIn('db_ro', flask.g)

    def test_read_helper_uses_replica_under_get(self):
        with self.app.test_request_context('/alerts', method='GET'):
            self.assertEqual(db._fetchall('SELECT 1 AS n', {})[0].n, 1)
            self.assertIn('db_ro', flask.g)

    def test_read_helper_uses_primary_under_post(self):
        with self.app.test_request_context('/alert', method='POST'):
            self.assertEqual(db._fetchall('SELECT 1 AS n', {})[0].n, 1)
            self.assertNotIn('db_ro', flask.g)

    def test_write_on_read_only_connection_is_rejected(self):
        import psycopg2

        with self.app.test_request_context('/alerts', method='GET'):
            conn = db.get_db(readonly=True)
            with self.assertRaises(psycopg2.errors.ReadOnlySqlTransaction):
                conn.cursor().execute('CREATE TABLE should_not_exist (id int)')

    def test_connection_is_closed_when_the_post_connect_setup_fails(self):
        import psycopg2

        # the connection is already open by then, so bailing out must not leak the socket
        class HalfDeadConnection:
            """Connects fine, then rejects the post-connect SETs (mid-promotion standby)."""

            def __init__(self):
                self.autocommit = False
                self.closed = False

            def set_client_encoding(self, enc):
                pass

            def cursor(self):
                raise psycopg2.OperationalError('terminating connection due to administrator command')

            def close(self):
                self.closed = True

        conn = HalfDeadConnection()

        with patch('alerta.database.backends.postgres.base.psycopg2.connect', lambda *a, **kw: conn):
            with self.assertRaises(psycopg2.OperationalError):
                db.connect_ro()

        self.assertTrue(conn.closed, 'the half-initialised connection was leaked')

    def test_zero_connect_timeout_is_passed_through(self):
        import psycopg2

        # 0 is falsy: it must still reach psycopg2 instead of silently dropping the timeout
        seen = {}
        real_connect = psycopg2.connect

        def capturing_connect(*args, **kwargs):
            seen.update(kwargs)
            return real_connect(*args, **kwargs)

        with patch('alerta.database.backends.postgres.base.psycopg2.connect', capturing_connect):
            db._connect(db.uri, connect_timeout=0).close()

        self.assertEqual(seen.get('connect_timeout'), 0)


class PostgresReplicaFallbackTestCase(unittest.TestCase):
    """
    A replica outage must degrade performance, not availability.
    """

    def setUp(self):
        self.app = create_app({'TESTING': True})
        if not self.app.config['DATABASE_URL'].startswith('postgres'):
            self.skipTest('Postgres backend only')
        self.app = create_app({
            'TESTING': True,
            # unroutable host: connecting fails fast and the breaker trips
            'DATABASE_URL_RO': 'postgresql://alerta@127.0.0.1:1/alerta',
            'DATABASE_URL_RO_CONNECT_TIMEOUT': 1,
            'DATABASE_URL_RO_RETRY_INTERVAL': 30,
        })
        db._replica_down_until = 0.0

    def tearDown(self):
        db._replica_down_until = 0.0
        db.destroy()

    def test_unreachable_replica_falls_back_to_primary(self):
        with self.app.test_request_context('/alerts', method='GET'):
            conn = db.get_db(readonly=True)
            self.assertIs(conn, db.get_db())          # served from the primary
            self.assertNotIn('db_ro', flask.g)
            self.assertEqual(db._fetchall('SELECT 1 AS n', {})[0].n, 1)

    def test_breaker_stops_further_connection_attempts(self):
        with self.app.test_request_context('/alerts', method='GET'):
            db.get_db(readonly=True)                   # first failure trips the breaker
        self.assertGreater(db._replica_down_until, time.monotonic())

        with patch.object(type(db), 'connect_ro') as mock_connect_ro:
            with self.app.test_request_context('/alerts', method='GET'):
                self.assertFalse(db._use_replica())    # breaker open: do not even try
                db.get_db(readonly=True)
            mock_connect_ro.assert_not_called()

    def test_is_replica_alive_is_false_while_the_breaker_is_open(self):
        with self.app.test_request_context('/alerts', method='GET'):
            db.get_db(readonly=True)                   # first failure trips the breaker
        self.assertFalse(db.is_replica_alive)
        response = self.app.test_client().get('/management/gtg')
        self.assertEqual(response.status_code, 200)    # degraded, not down
        self.assertEqual(response.data.decode(), 'OK REPLICA_DOWN')

    def test_breaker_closes_after_retry_interval(self):
        with self.app.test_request_context('/alerts', method='GET'):
            db.get_db(readonly=True)
        db._replica_down_until = time.monotonic() - 1  # simulate the interval elapsing

        with patch.object(type(db), 'connect_ro') as mock_connect_ro:
            with self.app.test_request_context('/alerts', method='GET'):
                self.assertTrue(db._use_replica())     # breaker closed: try again
                db.get_db(readonly=True)
            mock_connect_ro.assert_called_once()


class PostgresReplicaMidQueryTestCase(unittest.TestCase):
    """
    A replica that dies mid-query: the same SELECT is retried on the primary.
    Errors raised by the PRIMARY keep propagating untouched (zero regression).
    """

    def setUp(self):
        self.app = create_app({'TESTING': True})
        if not self.app.config['DATABASE_URL'].startswith('postgres'):
            self.skipTest('Postgres backend only')
        uri = self.app.config['DATABASE_URL']
        self.app = create_app({'TESTING': True, 'DATABASE_URL_RO': uri + '?application_name=alerta_ro'})
        db._replica_down_until = 0.0

    def tearDown(self):
        db._replica_down_until = 0.0
        db.destroy()

    def test_replica_error_retries_on_primary(self):
        import psycopg2

        real_execute = type(db)._execute

        def flaky_execute(self, conn, query, vars):
            if conn is flask.g.get('db_ro'):
                raise psycopg2.OperationalError('server closed the connection unexpectedly')
            return real_execute(self, conn, query, vars)

        with self.app.test_request_context('/alerts', method='GET'):
            with patch.object(type(db), '_execute', flaky_execute):
                rows = db._fetchall('SELECT 1 AS n', {})
            self.assertEqual(rows[0].n, 1)             # answered by the primary
            self.assertNotIn('db_ro', flask.g)         # replica dropped for this request
        self.assertGreater(db._replica_down_until, time.monotonic())  # breaker tripped

    def _raise_from_replica(self, exc):
        """Patch _execute so the replica raises `exc` and the primary answers normally."""
        real_execute = type(db)._execute

        def failing_execute(backend, conn, query, vars):
            if conn is flask.g.get('db_ro'):
                raise exc
            return real_execute(backend, conn, query, vars)

        return patch.object(type(db), '_execute', failing_execute)

    def test_recovery_conflict_retries_without_tripping_breaker(self):
        # a recovery conflict is 40001, or 40P01 for the buffer-deadlock case -- not 57014
        import psycopg2

        for sqlstate in ('40001', '40P01'):
            with self.subTest(sqlstate=sqlstate):
                db._replica_down_until = 0.0

                class RecoveryConflict(psycopg2.OperationalError):
                    pgcode = sqlstate

                with self.app.test_request_context('/alerts', method='GET'):
                    with self._raise_from_replica(
                            RecoveryConflict('canceling statement due to conflict with recovery')):
                        rows = db._fetchall('SELECT 1 AS n', {})
                    self.assertEqual(rows[0].n, 1)      # answered by the primary
                # the replica is healthy: the breaker must stay closed
                self.assertLessEqual(db._replica_down_until, time.monotonic())

    def test_statement_timeout_is_not_retried_on_the_primary(self):
        # 57014 is a deliberate cancel on a HEALTHY replica: re-running it on the primary
        # would move the shed load back onto the node this feature protects
        import psycopg2

        class QueryCanceled(psycopg2.OperationalError):
            pgcode = '57014'

        primary_calls = []
        real_execute = type(db)._execute

        def failing_execute(backend, conn, query, vars):
            if conn is flask.g.get('db_ro'):
                raise QueryCanceled('canceling statement due to statement timeout')
            primary_calls.append(conn)
            return real_execute(backend, conn, query, vars)

        with self.app.test_request_context('/alerts', method='GET'):
            with patch.object(type(db), '_execute', failing_execute):
                with self.assertRaises(QueryCanceled):
                    db._fetchall('SELECT 1 AS n', {})
        self.assertEqual(primary_calls, [], 'the cancelled query was re-run on the primary')
        self.assertLessEqual(db._replica_down_until, time.monotonic())  # breaker stays closed

    def test_primary_error_propagates_without_retry(self):
        import psycopg2

        calls = []

        def failing_execute(self, conn, query, vars):
            calls.append(conn)
            raise psycopg2.OperationalError('primary is gone')

        # POST => no replica, the failing connection IS the primary
        with self.app.test_request_context('/alert', method='POST'):
            with patch.object(type(db), '_execute', failing_execute):
                with self.assertRaises(psycopg2.OperationalError):
                    db._fetchall('SELECT 1 AS n', {})
        self.assertEqual(len(calls), 1, 'a primary error must not be retried')


class GoodToGoReplicaTestCase(unittest.TestCase):
    """
    `/management/gtg` reports the replica without failing on it.

    The endpoint is normally wired to a readiness probe, so a 503 pulls the node out of
    rotation. A down replica costs the offload, not availability -- the primary still
    serves every request -- so failing the probe would turn a degradation into an outage.
    """

    def setUp(self):
        self.app = create_app({'TESTING': True})

    def tearDown(self):
        db.destroy()

    def test_ok_when_no_replica_is_configured(self):
        # feature off: "not offloading" must never read as a fault
        self.assertTrue(db.is_replica_alive)
        response = self.app.test_client().get('/management/gtg')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), 'OK')

    def test_down_replica_reports_but_does_not_fail_the_probe(self):
        with patch.object(type(db), 'is_replica_alive', property(lambda self: False)):
            response = self.app.test_client().get('/management/gtg')
        self.assertEqual(response.status_code, 200, 'a down replica must not fail readiness')
        self.assertEqual(response.data.decode(), 'OK REPLICA_DOWN')

    def test_down_primary_still_fails(self):
        with patch.object(type(db), 'is_alive', property(lambda self: False)):
            response = self.app.test_client().get('/management/gtg')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data.decode(), 'FAILED')


class AuthReadsPinnedToPrimaryTestCase(unittest.TestCase):
    """
    Authentication and authorization reads must never be served by a replica.

    Revocation is a write, so it is strongly consistent; it is the *enforcement* read that
    would go stale, keeping a revoked key, scope or customer mapping working for the lag.
    """

    AUTH_READS = ('get_key', 'get_scopes_by_match', 'get_customers_by_match',
                  'get_user_by_username')

    def setUp(self):
        self.app = create_app({'TESTING': True})
        uri = self.app.config['DATABASE_URL']
        if uri.startswith('postgres'):
            overrides = {'DATABASE_URL_RO': uri + '?application_name=alerta_ro'}
        else:
            overrides = {'DATABASE_READ_PREFERENCE': 'secondaryPreferred'}
        self.app = create_app({'TESTING': True, **overrides})

    def tearDown(self):
        db.destroy()

    def _call(self, name):
        # these raise when nothing matches; the read still happened, which is the point
        from alerta.exceptions import NoCustomerMatch
        try:
            return self._invoke(name)
        except NoCustomerMatch:
            return None

    def _invoke(self, name):
        return {
            'get_key': lambda: db.get_key('no-such-key'),
            'get_scopes_by_match': lambda: db.get_scopes_by_match('nobody', matches=['nobody']),
            'get_customers_by_match': lambda: db.get_customers_by_match('nobody', matches=['nobody']),
            'get_user_by_username': lambda: db.get_user_by_username('nobody'),
        }[name]()

    def test_auth_reads_never_touch_the_replica(self):
        # gate wide open: a GET on the api blueprint with the feature enabled
        for name in self.AUTH_READS:
            with self.subTest(read=name):
                with self.app.test_request_context('/alerts', method='GET'):
                    self.assertTrue(db._use_replica(), 'precondition: the gate must be open')
                    self._call(name)
                    self.assertNotIn(
                        'db_ro', flask.g,
                        f'{name}() opened a replica handle for an auth decision')

    def test_a_plain_data_read_still_uses_the_replica(self):
        # guards the test above against passing because routing is broken altogether
        with self.app.test_request_context('/alerts', method='GET'):
            db.get_alerts(query=None, page=1, page_size=1)
            self.assertIn('db_ro', flask.g, 'ordinary reads should still be offloaded')


class MongoReplicaTestCase(unittest.TestCase):
    """
    Mongo routing. CI runs a standalone server, where read preference changes no actual
    routing: what is asserted here is the handle options and the gate, not the target node.
    """

    def setUp(self):
        self.app = create_app({'TESTING': True})
        if not self.app.config['DATABASE_URL'].startswith('mongodb'):
            self.skipTest('MongoDB backend only')
        self.app = create_app({'TESTING': True, 'DATABASE_READ_PREFERENCE': 'secondaryPreferred'})

    def tearDown(self):
        db.destroy()

    def test_get_uses_secondary_preferred_handle(self):
        from pymongo import ReadPreference

        with self.app.test_request_context('/alerts', method='GET'):
            handle = db.get_db()
            self.assertIn('db_ro', flask.g)
            self.assertEqual(handle.read_preference, ReadPreference.SECONDARY_PREFERRED)

    def test_post_uses_primary_handle(self):
        from pymongo import ReadPreference

        with self.app.test_request_context('/alert', method='POST'):
            handle = db.get_db()
            self.assertNotIn('db_ro', flask.g)
            self.assertEqual(handle.read_preference, ReadPreference.PRIMARY)

    def test_invalid_read_preference_falls_back_to_primary(self):
        from pymongo import ReadPreference

        app = create_app({'TESTING': True, 'DATABASE_READ_PREFERENCE': 'not-a-preference'})
        self.assertEqual(db.read_preference, ReadPreference.PRIMARY)
        with app.test_request_context('/alerts', method='GET'):
            self.assertFalse(db._use_replica())
            self.assertNotIn('db_ro', flask.g)


if __name__ == '__main__':
    unittest.main()
