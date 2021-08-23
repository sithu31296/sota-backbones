from tabulate import tabulate

import sys
sys.path.insert(0, '.')
import models


def main():
    model_names = list(models.__all__.keys())
    model_variants = []
    for name in model_names:
        model_variants.append(list(eval(f'models.{name}_settings').keys()))

    print(tabulate({'Model Names': model_names, 'Model Variants': model_variants}, headers='keys'))

if __name__ == '__main__':
    main()