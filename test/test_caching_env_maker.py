from unittest import TestCase, skip
from models.caching_environment_maker import CachingEnvironmentMaker, GVGAI_RUBEN, GVGAI_BAN4D

class CachingEnvMakerTest(TestCase):
    @skip('not currently testing')
    def test_ruben(self):
        EnvMaker = CachingEnvironmentMaker(GVGAI_RUBEN)
        for _ in range(100):
            for lvl in range(5):
                env = EnvMaker.make('gvgai-aliens-lvl{}-v0'.format(lvl))
                self.assertIsNotNone(env)

    def test_ban4d(self):
        EnvMaker = CachingEnvironmentMaker(GVGAI_BAN4D)
        for _ in range(100):
            for lvl in range(5):
                env = EnvMaker.make('gvgai-aliens-lvl{}-v0'.format(lvl))
                self.assertIsNotNone(env)
