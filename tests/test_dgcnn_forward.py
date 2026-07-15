import unittest

import torch

from src.models.dgcnn import DGCNNSegmentation, knn


class TestDGCNNForward(unittest.TestCase):
    def test_forward_shape_and_groupnorm(self) -> None:
        torch.manual_seed(42)
        model = DGCNNSegmentation(
            input_channels=6,
            num_classes=17,
            k=8,
            emb_dims=64,
            dropout=0.0,
        )
        x = torch.randn(2, 128, 6)

        group_norms = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.GroupNorm)
        ]
        batch_norms = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        ]
        model.train()
        with torch.no_grad():
            train_logits = model(x)
        model.eval()
        with torch.no_grad():
            eval_logits = model(x)

        self.assertEqual(train_logits.shape, (2, 128, 17))
        self.assertEqual(len(group_norms), 8)
        self.assertEqual(batch_norms, [])
        torch.testing.assert_close(train_logits, eval_logits)

    def test_chunked_knn_matches_full_computation(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(2, 6, 37)

        expected = _full_knn(x, k=8)
        actual = knn(x, k=8, chunk_size=7)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _full_knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    norm = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -norm - inner - norm.transpose(2, 1)
    return pairwise_distance.topk(k=min(k, x.shape[-1]), dim=-1)[1]


if __name__ == "__main__":
    unittest.main()
