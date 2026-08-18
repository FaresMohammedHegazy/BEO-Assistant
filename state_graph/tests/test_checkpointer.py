import os
import tempfile
import unittest

from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint

from state_graph.checkpointer import get_checkpointer


class TestSqliteCheckpointer(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        fd, self.tmp_db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)

    def tearDown(self):
        os.remove(self.tmp_db_path)

    async def test_write_and_read_dummy_state(self):
        config = {
        "configurable": {
            "thread_id": "test-thread-issue-2",
            "checkpoint_ns": "",
            }
        }

        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {"status": "DRAFT", "event_id": "EVT_999"}

        metadata = CheckpointMetadata(source="input", step=1, writes={}, parents={})

        async with get_checkpointer(self.tmp_db_path) as checkpointer:
            await checkpointer.aput(config, checkpoint, metadata, {})
            result = await checkpointer.aget_tuple(config)

        self.assertIsNotNone(result)
        self.assertEqual(result.checkpoint["channel_values"]["status"], "DRAFT")
        self.assertEqual(result.checkpoint["channel_values"]["event_id"], "EVT_999")


if __name__ == "__main__":
    unittest.main()