import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from envelope_mappings import (
    AmbiguousMatch,
    CompanyLogo,
    EnvelopeClassifier,
    EnvelopeTemplate,
    NewTemplateNeeded,
    PatternRecord,
)
from envelope_mappings.fingerprint import EnvelopeFingerprint


def _make_logo(seed: int) -> np.ndarray:
    """A synthetic 'logo' with enough texture for ORB to find keypoints."""
    rng = np.random.RandomState(seed)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    for _ in range(20):
        pt1 = tuple(rng.randint(0, 100, 2))
        pt2 = tuple(rng.randint(0, 100, 2))
        cv2.line(img, pt1, pt2, (0, 0, 0), 2)
    return img


def _make_envelope(seed: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.ones((400, 300, 3), dtype=np.uint8)
    img[:] = color
    rng = np.random.RandomState(seed)
    for _ in range(10):
        pt1 = tuple(rng.randint(0, 300, 2))
        pt2 = tuple(rng.randint(0, 300, 2))
        cv2.line(img, pt1, pt2, (0, 0, 0), 2)
    cv2.putText(
        img, "PATTERN 1234", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )
    return img


def _make_logo_at(path: Path, seed: int) -> None:
    cv2.imwrite(str(path), _make_logo(seed))


def _make_envelope_at(path: Path, seed: int, color: tuple[int, int, int]) -> None:
    cv2.imwrite(str(path), _make_envelope(seed, color))


class Vogue1970(EnvelopeTemplate):
    def extract_fields(self, envelope):
        return {"pattern_number": "1234"}


class KwikSew1985(EnvelopeTemplate):
    company = "KwikSew"

    def extract_fields(self, envelope):
        return {}


class _FixedDistanceFingerprint:
    """Stub fingerprint whose distance() always returns a pre-set score,
    for deterministically testing threshold branching without needing to
    tune real images into an exact confidence window.
    """

    def __init__(self, score: float):
        self._score = score

    def distance(self, other) -> float:
        return self._score


class _FixedScoreTemplate(EnvelopeTemplate):
    """A template whose match score against ANY envelope is fixed at
    construction time -- see _FixedDistanceFingerprint.
    """

    def __init__(self, company: str, year_code: int, score: float):
        self.company = company
        self.year_code = year_code
        super().__init__()
        self._reference_fingerprint = _FixedDistanceFingerprint(score)

    def fingerprint(self, envelope):
        return self._reference_fingerprint  # distance() ignores its argument

    def extract_fields(self, envelope):
        return {}


def _make_vogue_logo(ref_image: np.ndarray) -> CompanyLogo:
    logo = CompanyLogo("Vogue")
    logo.set_reference(ref_image)
    return logo


class TemplateClassnameParsingTests(unittest.TestCase):
    def test_single_word_company_and_year_auto_derive(self):
        t = Vogue1970()
        self.assertEqual(t.company, "Vogue")
        self.assertEqual(t.year_code, 1970)

    def test_two_word_company_needs_explicit_override(self):
        t = KwikSew1985()
        self.assertEqual(t.company, "KwikSew")
        self.assertEqual(t.year_code, 1985)  # year still auto-derives fine

    def test_unparseable_classname_falls_back_gracefully(self):
        class NoYearHere(EnvelopeTemplate):
            def extract_fields(self, envelope):
                return {}

        t = NoYearHere()
        self.assertEqual(t.company, "NoYearHere")
        self.assertIsNone(t.year_code)

    def test_year_range_is_optional_and_independent_of_year_code(self):
        class Pictorial1934(EnvelopeTemplate):
            year_range = (1932, 1936)

            def extract_fields(self, envelope):
                return {}

        t = Pictorial1934()
        self.assertEqual(t.year_code, 1934)  # still auto-derives as normal
        self.assertEqual(t.year_range, (1932, 1936))  # explicit, separate

    def test_year_range_defaults_to_none_when_not_set(self):
        t = Vogue1970()
        self.assertIsNone(t.year_range)


class FingerprintTests(unittest.TestCase):
    def test_identical_images_score_near_one(self):
        img = _make_envelope(1, (200, 200, 240))
        fp1 = EnvelopeFingerprint.compute(img)
        fp2 = EnvelopeFingerprint.compute(img)
        self.assertGreater(fp1.distance(fp2), 0.99)

    def test_very_different_images_score_lower(self):
        img_a = _make_envelope(1, (200, 200, 240))
        img_b = _make_envelope(99, (30, 30, 30))
        fp_a = EnvelopeFingerprint.compute(img_a)
        fp_b = EnvelopeFingerprint.compute(img_b)
        self.assertLess(fp_a.distance(fp_b), fp_a.distance(fp_a))


class CompanyLogoTests(unittest.TestCase):
    def test_matching_logo_scores_high(self):
        ref = _make_logo(1)
        logo = _make_vogue_logo(ref)
        self.assertGreater(logo.match_score(ref), 0.9)

    def test_blank_region_scores_zero(self):
        ref = _make_logo(1)
        logo = _make_vogue_logo(ref)
        blank = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.assertEqual(logo.match_score(blank), 0.0)

    def test_match_score_without_any_reference_raises(self):
        logo = CompanyLogo("Vogue")  # neither set_reference nor path called
        with self.assertRaises(ValueError):
            logo.match_score(_make_logo(1))


class LazyReferencePathTests(unittest.TestCase):
    """Covers the actual new behavior: pointing a template/logo at a path
    before the file exists there, then having it load correctly once the
    file shows up -- this is the real-world workflow (reference images
    live outside the repo, placed wherever the person keeps them).
    """

    def test_template_path_set_before_file_exists_raises_on_access(self):
        with TemporaryDirectory() as tmpdir:
            not_yet_there = Path(tmpdir) / "vogue_1970.jpg"
            t = Vogue1970()
            t.set_reference_path(not_yet_there)
            with self.assertRaises(FileNotFoundError):
                _ = t.reference_fingerprint

    def test_template_path_works_once_file_is_placed(self):
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "vogue_1970.jpg"
            t = Vogue1970()
            t.set_reference_path(image_path)  # set BEFORE file exists

            _make_envelope_at(image_path, 1, (200, 200, 240))  # now place it

            fp = t.reference_fingerprint
            self.assertIsNotNone(fp)
            # cached -- second access doesn't re-raise or re-read
            self.assertIs(t.reference_fingerprint, fp)

    def test_logo_path_set_before_file_exists_raises_on_use(self):
        with TemporaryDirectory() as tmpdir:
            not_yet_there = Path(tmpdir) / "vogue_logo.jpg"
            logo = CompanyLogo("Vogue")
            logo.set_reference_path(not_yet_there)
            with self.assertRaises(FileNotFoundError):
                logo.match_score(_make_logo(1))

    def test_logo_path_works_once_file_is_placed(self):
        with TemporaryDirectory() as tmpdir:
            logo_path = Path(tmpdir) / "vogue_logo.jpg"
            logo = CompanyLogo("Vogue")
            logo.set_reference_path(logo_path)  # set BEFORE file exists

            _make_logo_at(logo_path, 1)  # now place it

            ref = cv2.imread(str(logo_path))
            self.assertGreater(logo.match_score(ref), 0.9)

    def test_classifier_skips_template_with_missing_reference_file_with_warning(self):
        with TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "does_not_exist.jpg"
            t = Vogue1970()
            t.set_reference_path(missing_path)

            vogue_logo_ref = _make_logo(1)
            logo = _make_vogue_logo(vogue_logo_ref)
            classifier = EnvelopeClassifier(logos=[logo], templates=[t])

            matching_envelope = _make_envelope(1, (200, 200, 240))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = classifier.classify(
                    matching_envelope, logo_region=vogue_logo_ref
                )
            self.assertIsInstance(result, NewTemplateNeeded)
            self.assertTrue(any("Vogue1970" in str(w.message) for w in caught))

    def test_classifier_skips_logo_with_missing_reference_file_with_warning(self):
        with TemporaryDirectory() as tmpdir:
            missing_path = Path(tmpdir) / "does_not_exist.jpg"
            logo = CompanyLogo("Vogue")
            logo.set_reference_path(missing_path)

            classifier = EnvelopeClassifier(logos=[logo], templates=[])
            matching_envelope = _make_envelope(1, (200, 200, 240))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = classifier.classify(matching_envelope)
            self.assertIsInstance(result, NewTemplateNeeded)
            self.assertEqual(result.stage, "company")
            self.assertTrue(any("Vogue" in str(w.message) for w in caught))


class EnvelopeClassifierTests(unittest.TestCase):
    def setUp(self):
        self.vogue_logo_ref = _make_logo(1)
        self.vogue_logo = _make_vogue_logo(self.vogue_logo_ref)

        self.vogue_template = Vogue1970()
        self.vogue_reference_envelope = _make_envelope(1, (200, 200, 240))
        self.vogue_template.set_reference(self.vogue_reference_envelope)

        self.classifier = EnvelopeClassifier(
            logos=[self.vogue_logo], templates=[self.vogue_template]
        )

    def test_matching_envelope_and_logo_yields_confident_record(self):
        matching_envelope = _make_envelope(1, (200, 200, 240))
        result = self.classifier.classify(
            matching_envelope, logo_region=self.vogue_logo_ref
        )
        self.assertIsInstance(result, PatternRecord)
        self.assertEqual(result.company, "Vogue")
        self.assertEqual(result.year_code, 1970)
        self.assertEqual(result.fields, {"pattern_number": "1234"})

    def test_unrelated_logo_region_yields_new_template_needed_at_company_stage(self):
        matching_envelope = _make_envelope(1, (200, 200, 240))
        blank_region = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = self.classifier.classify(matching_envelope, logo_region=blank_region)
        self.assertIsInstance(result, NewTemplateNeeded)
        self.assertEqual(result.stage, "company")

    def test_logo_matches_but_no_templates_for_that_company(self):
        classifier = EnvelopeClassifier(logos=[self.vogue_logo], templates=[])
        matching_envelope = _make_envelope(1, (200, 200, 240))
        result = classifier.classify(matching_envelope, logo_region=self.vogue_logo_ref)
        self.assertIsInstance(result, NewTemplateNeeded)
        self.assertEqual(result.stage, "template")

    def test_template_without_reference_set_does_not_crash_classification(self):
        unfinished = Vogue1970()  # set_reference() never called
        classifier = EnvelopeClassifier(logos=[self.vogue_logo], templates=[unfinished])
        matching_envelope = _make_envelope(1, (200, 200, 240))
        result = classifier.classify(matching_envelope, logo_region=self.vogue_logo_ref)
        # Scored 0 against the unfinished template -- falls through to
        # NewTemplateNeeded rather than raising.
        self.assertIsInstance(result, NewTemplateNeeded)

    def test_no_logos_registered_yields_new_template_needed(self):
        classifier = EnvelopeClassifier(logos=[], templates=[self.vogue_template])
        matching_envelope = _make_envelope(1, (200, 200, 240))
        result = classifier.classify(matching_envelope)
        self.assertIsInstance(result, NewTemplateNeeded)
        self.assertEqual(result.stage, "company")

    def test_score_between_thresholds_yields_ambiguous_match(self):
        # Tuning real images to land in an exact confidence window is
        # fragile -- a stub with a fixed, controllable score tests the
        # threshold branching itself directly and deterministically.
        mid_score = (
            EnvelopeTemplate.LOW_THRESHOLD + EnvelopeTemplate.HIGH_THRESHOLD
        ) / 2
        stub_a = _FixedScoreTemplate("Vogue", 1971, mid_score)
        stub_b = _FixedScoreTemplate("Vogue", 1972, mid_score - 0.05)
        classifier = EnvelopeClassifier(
            logos=[self.vogue_logo], templates=[stub_a, stub_b]
        )
        matching_envelope = _make_envelope(1, (200, 200, 240))
        result = classifier.classify(
            matching_envelope, logo_region=self.vogue_logo_ref
        )
        self.assertIsInstance(result, AmbiguousMatch)
        self.assertEqual(result.company, "Vogue")
        self.assertEqual(len(result.candidates), 2)  # both stubs offered

    def test_score_below_low_threshold_yields_new_template_needed(self):
        stub = _FixedScoreTemplate("Vogue", 1971, EnvelopeTemplate.LOW_THRESHOLD - 0.1)
        classifier = EnvelopeClassifier(logos=[self.vogue_logo], templates=[stub])
        matching_envelope = _make_envelope(1, (200, 200, 240))
        result = classifier.classify(matching_envelope, logo_region=self.vogue_logo_ref)
        self.assertIsInstance(result, NewTemplateNeeded)
        self.assertEqual(result.stage, "template")


class ExtendViaAppendTests(unittest.TestCase):
    """Confirms the core design goal: adding coverage is purely additive,
    nothing about EnvelopeClassifier itself needs to change.
    """

    def test_appending_a_new_company_does_not_affect_existing_matches(self):
        vogue_logo_ref = _make_logo(1)
        vogue_logo = _make_vogue_logo(vogue_logo_ref)
        vogue_template = Vogue1970()
        vogue_template.set_reference(_make_envelope(1, (200, 200, 240)))

        logos = [vogue_logo]
        templates = [vogue_template]
        classifier = EnvelopeClassifier(logos, templates)

        # extend purely by appending -- classifier instance already holds
        # references to these same lists, no re-construction needed
        kwiksew_logo_ref = _make_logo(2)
        kwiksew_logo = CompanyLogo("KwikSew")
        kwiksew_logo.set_reference(kwiksew_logo_ref)
        logos.append(kwiksew_logo)

        kwiksew_template = KwikSew1985()
        kwiksew_template.set_reference(_make_envelope(2, (100, 150, 100)))
        templates.append(kwiksew_template)

        # original Vogue match still works after extension
        matching_envelope = _make_envelope(1, (200, 200, 240))
        result = classifier.classify(matching_envelope, logo_region=vogue_logo_ref)
        self.assertIsInstance(result, PatternRecord)
        self.assertEqual(result.company, "Vogue")

        # new KwikSew coverage works too, without touching the classifier
        kwiksew_envelope = _make_envelope(2, (100, 150, 100))
        result2 = classifier.classify(kwiksew_envelope, logo_region=kwiksew_logo_ref)
        self.assertIsInstance(result2, PatternRecord)
        self.assertEqual(result2.company, "KwikSew")


if __name__ == "__main__":
    unittest.main()
