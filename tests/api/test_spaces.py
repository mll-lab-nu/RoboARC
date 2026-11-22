"""The spaces (Gymnasium's when installed, else the package's stand-in, whose interface is tested here directly)."""
import numpy as np

from roboarc import spaces


def test_box_standin():
    b = spaces._Box(np.array([-1.0, 0.0]), np.array([1.0, 2.0]), dtype=np.float32, seed=0)
    assert b.shape == (2,) and b.dtype == np.float32
    for _ in range(20):
        assert b.contains(b.sample())
    assert not b.contains(np.array([2.0, 0.0], np.float32))
    assert not b.contains(np.zeros(3, np.float32))
    u = spaces._Box(0, 255, (4, 5, 3), np.uint8, seed=1)
    x = u.sample()
    assert x.dtype == np.uint8 and x.shape == (4, 5, 3) and u.contains(x)
    inf = spaces._Box(-np.inf, np.inf, (3,), np.float64, seed=2)
    assert np.all(np.isfinite(inf.sample()))


def test_dict_standin():
    d = spaces._Dict({"rgb": spaces._Dict({"cam": spaces._Box(0, 255, (2, 2, 3), np.uint8)}),
                      "proprio": spaces._Box(-np.inf, np.inf, (5,), np.float32)}, seed=3)
    s = d.sample()
    assert set(s) == {"rgb", "proprio"} and d.contains(s)
    assert list(d.keys()) == ["rgb", "proprio"] and d["rgb"]["cam"].shape == (2, 2, 3)
    s["proprio"] = np.zeros(4, np.float32)
    assert not d.contains(s)


def test_factories_match_gymnasium_when_present():
    b = spaces.Box(np.zeros(3), np.ones(3), (3,), np.float32)
    assert b.shape == (3,) and b.contains(np.full(3, 0.5, np.float32))
    if spaces.HAVE_GYMNASIUM:
        import gymnasium
        assert isinstance(b, gymnasium.spaces.Box) and issubclass(spaces.EnvBase, gymnasium.Env)
