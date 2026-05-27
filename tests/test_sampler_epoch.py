from datamodule.utils import BucketedBatchSampler


def test_bucketed_batch_sampler_changes_order_between_epochs():
    data = []
    for i in range(20):
        # (fname, (width, height), label_tokens)
        data.append((f"img_{i}", (10 + i, 20 + i), ["a", "b"]))

    sampler = BucketedBatchSampler(
        data=data,
        max_pixels_per_batch=10_000,
        max_batch_size=2,
        shuffle=True,
        maxlen=200,
        max_image_size=10_000,
        seed=7,
    )

    sampler.set_epoch(0)
    order0 = list(iter(sampler))

    sampler.set_epoch(1)
    order1 = list(iter(sampler))

    assert order0 != order1
