class RaptorEvaluator:
    def __init__(self, ground_truth, retrieved_results):
        self.ground_truth = ground_truth
        self.retrieved_results = retrieved_results

    def precision(self):
        true_positives = len(set(self.ground_truth) & set(self.retrieved_results))
        return true_positives / len(self.retrieved_results) if self.retrieved_results else 0

    def recall(self):
        true_positives = len(set(self.ground_truth) & set(self.retrieved_results))
        return true_positives / len(self.ground_truth) if self.ground_truth else 0

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

    def evaluate(self):
        return {
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1_score()
        }