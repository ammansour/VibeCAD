import unittest


class TestIntentRouter(unittest.TestCase):
    class _FakeResponse:
        def __init__(self, content: str):
            self.content = content

    class _FakeLLM:
        def __init__(self, content: str, available: bool = True):
            self.is_available = available
            self._content = content

        def chat(self, messages, system_prompt=None):
            return TestIntentRouter._FakeResponse(self._content)

    def test_llm_routes_to_qa(self):
        from vibecad.design.intent_router import decide_route

        llm = self._FakeLLM('{"route":"qa","reason":"question"}')
        out = decide_route(llm, "What resistor value do you recommend for R1?")
        self.assertEqual(out.route, "qa")

    def test_llm_routes_to_agent(self):
        from vibecad.design.intent_router import decide_route

        llm = self._FakeLLM('{"route":"agent","reason":"task"}')
        out = decide_route(llm, "Route the nets between U1 and R1")
        self.assertEqual(out.route, "agent")

    def test_fallback_question_mark(self):
        from vibecad.design.intent_router import decide_route

        llm = self._FakeLLM("", available=False)
        out = decide_route(llm, "What value for R1?")
        self.assertEqual(out.route, "qa")

    def test_fallback_default(self):
        from vibecad.design.intent_router import decide_route

        llm = self._FakeLLM("", available=False)
        out = decide_route(llm, "Place U1 near the connector")
        self.assertEqual(out.route, "agent")


if __name__ == "__main__":
    unittest.main()
