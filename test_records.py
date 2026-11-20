import unittest

from record_reader import read_records


class RecordsTest(unittest.TestCase):
    def setUp(self):
        self.tables = read_records()

    def references(self, left, left_key, right, right_key):
        candidates = {row[right_key] for row in self.tables[right]}
        self.assertTrue(all(row[left_key] in candidates for row in self.tables[left]))

    def test_documents_reference_source_records(self):
        self.references('events','plan','plans','plan')
        self.references('attendance','plan','plans','plan')
        self.references('capacity','mentor','mentors','mentor')
        self.assertEqual({row['term'] for row in self.tables['plans']}, set(range(1,7)))
        self.assertTrue(all(row['known_week'] >= row['event_week'] for row in self.tables['events']))
        self.assertTrue(any(row['known_week'] > row['event_week'] for row in self.tables['events']))
        self.assertTrue(any(row['state']=='accepted' for row in self.tables['attendance']))
        self.assertTrue(all(0 <= row['manual'] <= row['seats'] for row in self.tables['capacity']))
