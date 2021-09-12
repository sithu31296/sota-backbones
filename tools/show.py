from tabulate import tabulate

import sys
sys.path.insert(0, '.')
import models


def show_models():
    model_names = models.__all__
    model_variants = []
    for name in model_names:
        model_variants.append(list(eval(f'models.{name.lower()}_settings').keys()))

    print(tabulate({'Model Names': model_names, 'Model Variants': model_variants}, headers='keys'))

if __name__ == '__main__':
    show_models()