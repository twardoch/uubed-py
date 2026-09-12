# this_file: src/uubed/tmx.py
"""Stream explicit English/target translation pairs without normalizing their text."""

from collections.abc import Iterator
from pathlib import Path

from defusedxml.ElementTree import iterparse

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def language(value: str) -> str:
    """Canonicalize locale spelling, retaining regional/script distinctions."""
    result = value.strip().replace("_", "-").lower()
    if not result or not all(part.isascii() and part.isalnum() for part in result.split("-")):
        msg = "invalid language code"
        raise ValueError(msg)
    return result


def translation_pairs(path: Path) -> Iterator[dict[str, str]]:
    """Read English variants by xml:lang, never by segment position.

    Monolingual units and units without English have no translation pair and are
    skipped. Multiple English variants must agree or the source is ambiguous.
    Inline code text is retained; XML entity references are decoded by the parser.
    """
    with path.open("rb") as stream:
        context = iterparse(stream, events=("start", "end"))
        _, root = next(context)
        if root.tag != "tmx":
            msg = f"not a TMX document: {path}"
            raise ValueError(msg)
        ordinal = 0
        for event, element in context:
            if event != "end" or element.tag != "tu":
                continue
            ordinal += 1
            variants = []
            for variant in element.findall("tuv"):
                locale = language(variant.get(XML_LANG, variant.get("lang", "")))
                segment = variant.find("seg")
                if segment is None:
                    msg = f"missing segment in {path}, unit {ordinal}"
                    raise ValueError(msg)
                variants.append((locale, "".join(segment.itertext())))
            english = {text for locale, text in variants if locale.split("-")[0] == "en"}
            if len(english) > 1:
                msg = f"ambiguous English variants in {path}, unit {ordinal}"
                raise ValueError(msg)
            if english:
                source = english.pop()
                for locale, target in variants:
                    if locale.split("-")[0] != "en" and source.strip() and target.strip():
                        yield {
                            "source": source,
                            "target": target,
                            "language": locale,
                            "origin": str(path),
                            "unit": element.get("tuid", str(ordinal)),
                        }
            element.clear()
            # Clear completed units from their parent as well; iterparse alone
            # retains empty child elements and otherwise grows with corpus size.
            body = root.find("body")
            if body is not None:
                body.clear()
