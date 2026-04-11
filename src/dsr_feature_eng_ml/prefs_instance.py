"""Neutral instantiation point for the Preferences singleton."""

from dsr_feature_eng_ml.preferences import Preferences

# The ONLY place the singleton is instantiated
prefs = Preferences()
