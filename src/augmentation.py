import albumentations as A

class Augmentation:

    @staticmethod
    def build_augmentation(h_flip_prob, blur_prob, blur_limit):
        transform = A.compose(
            A.HorizontalFlip(p = h_flip_prob),
            A.Blur(blur_limit = blur_limit, p = blur_prob)
        )
        return transform
