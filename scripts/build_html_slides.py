"""Build a portable HTML slide deck from the collected measured evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / "outputs/two_ideas_slides"
    data = (folder / "slide_data.json").read_text(encoding="utf-8")
    json.loads(data)
    template = (ROOT / "scripts/templates/two_ideas_slides.html").read_text(encoding="utf-8")
    if template.count("__SLIDE_DATA__") != 1:
        raise ValueError("Expected one evidence slot")
    html = template.replace("__SLIDE_DATA__", data.replace("<", r"\u003c"))
    target = folder / "camera_magnetic_acceleration.html"
    target.write_text(html, encoding="utf-8")
    print(json.dumps(dict(path=str(target), bytes=target.stat().st_size, slides=13,
                         self_contained=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
