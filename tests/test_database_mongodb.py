import threading
import unittest
from unittest.mock import patch

from alerta.app import create_app, db


class MongoClientLifecycleTestCase(unittest.TestCase):
    """
    The MongoDB backend must keep ONE MongoClient per process and reuse it.

    The regression these guard: `client` used to be a plain attribute reassigned by every
    `connect()` on the shared `db` singleton, and `close()` closed whatever it pointed at.
    Two overlapping requests therefore closed the same client twice and leaked the other one's
    pool and monitor threads, growing without bound until the server ran out of file
    descriptors.
    """

    def setUp(self):
        self.app = create_app({'TESTING': True})
        if not self.app.config['DATABASE_URL'].startswith('mongodb'):
            self.skipTest('MongoDB backend only')

    def tearDown(self):
        db.destroy()

    def test_client_is_reused_across_requests(self):
        seen = []
        for _ in range(5):
            with self.app.app_context():
                seen.append(id(db.connect().client))
        self.assertEqual(len(set(seen)), 1, 'a new MongoClient was created per request')

    def test_close_does_not_close_the_client(self):
        with self.app.app_context():
            conn = db.connect()
            client = conn.client
            db.close(conn)
        # the pool must still be usable: close() on an app context is a no-op
        self.assertEqual(client.admin.command('ping')['ok'], 1.0)
        self.assertIs(db.client, client)

    def test_overlapping_requests_share_one_client(self):
        # the exact shape that used to leak: N requests inside connect() at the same time
        n = 8
        barrier = threading.Barrier(n)
        seen = []
        lock = threading.Lock()

        def worker():
            with self.app.app_context():
                conn = db.connect()
                barrier.wait()
                with lock:
                    seen.append(id(conn.client))
                db.close(conn)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(seen), n)
        self.assertEqual(len(set(seen)), 1, f'{len(set(seen))} clients created for {n} requests')
        self.assertEqual(db.client.admin.command('ping')['ok'], 1.0)

    def test_client_is_rebuilt_in_a_new_process(self):
        # a client built in a preforking WSGI master must never be used by a worker:
        # pymongo documents that as a deadlock risk. The cache is keyed by PID.
        with self.app.app_context():
            parent = db.client

        real_getpid = __import__('os').getpid

        with patch('alerta.database.backends.mongodb.base.os.getpid',
                   lambda: real_getpid() + 1):
            with self.app.app_context():
                child = db.client

        self.assertIsNot(child, parent, 'the inherited client was reused after a fork')

    def test_client_is_rebuilt_when_the_uri_changes(self):
        with self.app.app_context():
            first = db.client

        original_uri = db.uri
        try:
            db.uri = original_uri + ('&' if '?' in original_uri else '?') + 'appname=alerta_other'
            with self.app.app_context():
                second = db.client
            self.assertIsNot(second, first, 'a stale client was reused after the URI changed')
        finally:
            db.uri = original_uri

    def test_disconnect_closes_and_allows_a_fresh_client(self):
        with self.app.app_context():
            first = db.client
        db.disconnect()
        with self.app.app_context():
            second = db.client
        self.assertIsNot(second, first)
        self.assertEqual(second.admin.command('ping')['ok'], 1.0)


if __name__ == '__main__':
    unittest.main()
