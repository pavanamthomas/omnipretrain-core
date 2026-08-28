from __future__ import annotations

from data.curriculum import CurriculumEngine, NGramReference, inspect_noise, script_family


def test_easy_before_hard() -> None:
    ref = [
        "A red bicycle leans against a brick wall at dusk.",
        "Two people sit on a bench facing a small lake.",
        "A calico cat sleeps on a stack of newspapers.",
    ]
    scorer = NGramReference().fit(ref)
    engine = CurriculumEngine(scorer)
    easy = "A red bicycle leans against a brick wall at dusk."
    hard = "xqz vrml fhtagn 0xdeadbeef <div><div><div>aaaa"
    ranked = engine.rank([hard, easy])
    assert ranked[0].text == easy
    assert ranked[0].score < ranked[1].score


def test_html_residue_penalised() -> None:
    clean = inspect_noise("A quiet street in the late afternoon.")
    dirty = inspect_noise("<div><span><p>A quiet street</p></span></div>" * 6)
    assert dirty.html_residue > clean.html_residue
    assert dirty.penalty() > clean.penalty()


def test_step_schedule_grows() -> None:
    scorer = NGramReference().fit(["hello world", "the cat sat"])
    engine = CurriculumEngine(scorer, pace="step", step_cuts=(0.33, 0.66))
    texts = [f"hello world {i}" for i in range(12)]
    ranked = engine.rank(texts)
    e0 = engine.schedule(ranked, epoch=0, max_epoch=9)
    e8 = engine.schedule(ranked, epoch=8, max_epoch=9)
    assert 0 < len(e0) < len(e8) <= 12


def test_empty_ref_raises() -> None:
    try:
        NGramReference().fit([])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_blank_image_ranks_harder() -> None:
    import torch

    from data.curriculum import inspect_image_noise

    scorer = NGramReference().fit(["A red bicycle leans against a brick wall at dusk."])
    engine = CurriculumEngine(scorer)
    caption = "A red bicycle leans against a brick wall at dusk."
    clean = torch.rand(3, 32, 32)
    blank = torch.zeros(3, 32, 32)
    ranked = engine.rank([caption, caption], images=[clean, blank])
    assert inspect_image_noise(blank).blank > inspect_image_noise(clean).blank
    assert ranked[0].image_noise is not None and ranked[1].image_noise is not None
    assert ranked[1].score >= ranked[0].score
    assert ranked[1].image_noise.blank >= ranked[0].image_noise.blank


def test_cjk_not_exiled_by_english_ngram() -> None:
    ref = [
        "A red bicycle leans against a brick wall at dusk.",
        "Two people sit on a bench facing a small lake.",
        "A calico cat sleeps on a stack of newspapers.",
        "Close-up of a ceramic bowl filled with rice and scallions.",
        "The storefront has gold lettering and a striped awning.",
    ]
    scorer = NGramReference().fit(ref)
    assert scorer._ref_script == "latin"
    engine = CurriculumEngine(scorer)
    easy = "A red bicycle leans against a brick wall at dusk."
    cjk = "赤い自転車が夕暮れのレンガ壁に立てかけてある。二人の人が小さな湖の前のベンチに座っている。"
    ocr = "xqz vrml fhtagn 0xdeadbeef <div><div><div>aaaa"
    assert script_family(cjk) == "cjk"
    ranked = engine.rank([ocr, cjk, easy])
    by = {s.text: s for s in ranked}
    # CJK must not sit past OCR just because every codepoint is unseen
    assert ranked[-1].text == ocr
    assert by[cjk].log_ppl < by[ocr].log_ppl
    assert abs(by[cjk].log_ppl - scorer._cross_script_ppl) < 1e-9

