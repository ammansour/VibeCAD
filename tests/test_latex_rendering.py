import unittest


class TestBasicLatexRendering(unittest.TestCase):
    def test_inline_math_renders_ohm_and_subscripts(self):
        from vibecad.ui.markdown_utils import render_basic_latex

        s = "- LED: $R = (V_{supply} - V_f)/I$ -> 260\\Omega"
        out = render_basic_latex(s)

        self.assertNotIn("$", out)
        self.assertIn("Ω", out)
        self.assertIn("V", out)

    def test_mu_and_times(self):
        from vibecad.ui.markdown_utils import render_basic_latex

        s = "Cap: $10\\mu F$ and $2\\times 3$"
        out = render_basic_latex(s)
        self.assertIn("10µ F", out)
        self.assertIn("2× 3", out)

    def test_frac(self):
        from vibecad.ui.markdown_utils import render_basic_latex

        s = "Gain: $\\frac{1}{2}$"
        out = render_basic_latex(s)
        self.assertIn("(1)/(2)", out)

    def test_code_fence_is_not_modified(self):
        from vibecad.ui.markdown_utils import render_basic_latex

        s = """Here:
```python
x = '$R=1$'
```
And $R=2$.
"""
        out = render_basic_latex(s)
        # Inside code fence should keep $...$
        self.assertIn("'$R=1$'", out)
        # Outside should be converted
        self.assertIn("And R=2.", out)

    def test_inline_code_is_not_modified(self):
        from vibecad.ui.markdown_utils import render_basic_latex

        s = "Use `$R=1$` literally, but $R=2$ is math."
        out = render_basic_latex(s)
        self.assertIn("`$R=1$`", out)
        self.assertIn("but R=2 is math", out)

    def test_html_path_uses_sub_tag_for_unsupported_subscripts(self):
        from vibecad.ui.markdown_utils import markdown_to_html_fragment

        frag = markdown_to_html_fragment("Value: $V_f$ and $x_{foo}$")
        self.assertIn("V<sub>f</sub>", frag)
        self.assertIn("x<sub>foo</sub>", frag)


if __name__ == "__main__":
    unittest.main()
