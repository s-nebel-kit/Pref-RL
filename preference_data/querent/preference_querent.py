from abc import ABC, abstractmethod

from preference_data.querent.oracle import AbstractOracle, RewardMaximizingOracle
from preference_data.query_selection.query_selector import AbstractQuerySelector, MostRecentlyGeneratedQuerySelector


class AbstractPreferenceQuerent(AbstractQuerySelector, ABC):

    def __init__(self, preferences, query_candidates):
        self.query_candidates = query_candidates
        self.preferences = preferences

    @abstractmethod
    def query_preferences(self, num_preferences):
        pass


class AbstractSyntheticPreferenceQuerent(AbstractPreferenceQuerent, AbstractOracle, ABC):

    def query_preferences(self, num_preferences):
        queries = self.select_queries(self.query_candidates, num_queries=num_preferences)
        self.preferences.extend([(query, self.answer(query)) for query in queries])


class SyntheticPreferenceQuerent(AbstractSyntheticPreferenceQuerent,
                                 MostRecentlyGeneratedQuerySelector, RewardMaximizingOracle):
    pass

class AbstractAsynchronousPreferenceQuerent(AbstractPreferenceQuerent, ABC):

    def __init__(self, preferences, query_candidates, segment_renderer, database, env):
        self.query_candidates = query_candidates
        self.preferences = preferences
        self.segment_renderer = segment_renderer
        self.database = database
        self.env = env

    @abstractmethod
    def fetch_preferences(self, num_preferences):
        pass

class AsynchronousPreferenceQuerent(AbstractAsynchronousPreferenceQuerent, MostRecentlyGeneratedQuerySelector):

    def query_preferences(self, num_preferences):
        queries = self.select_queries(self.query_candidates, num_queries=num_preferences)
        for query in queries:
            for segment in query:
                self.segment_renderer.render_segment(segment=segment)

    def fetch_preferences(self, num_preferences):
        pass