
import argparse
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
args = parser.parse_args()
print(args.poly)
