from hypothesis import settings


def test_hypothesis_env():
    pn = settings.get_current_profile_name()
    print("active profile:", pn)
    prof = settings.get_profile(pn)

    print("hypothesis deadline:", prof.deadline)
    print("profiles:", list(settings._profiles.keys()))
