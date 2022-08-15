from anesthetic.weighted_pandas import WeightedSeries as _WeightedSeries

class WeightedSeries(_WeightedSeries):
    _metadata = _WeightedSeries._metadata + ['label']

    def __init__(self, *args, **kwargs):
        label = kwargs.pop('label', None)
        super().__init__(*args, **kwargs)
        self.label = label

    @property
    def _constructor(self):
        return WeightedSeries
    
