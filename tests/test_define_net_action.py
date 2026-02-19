import pytest

from vibecad.design.design_agent import DesignAgent, DesignActionType


def test_get_action_handler_does_not_crash_with_define_net_registered():
    """Regression: _get_action_handler builds a handler map.

    If DEFINE_NET is registered but the method is missing, this will raise
    AttributeError and break *all* action execution.
    """
    agent = DesignAgent(llm_client=None)

    # Should not raise.
    handler = agent._get_action_handler(DesignActionType.ADD_COMPONENT)
    assert callable(handler)

    handler_define = agent._get_action_handler(DesignActionType.DEFINE_NET)
    assert callable(handler_define)
