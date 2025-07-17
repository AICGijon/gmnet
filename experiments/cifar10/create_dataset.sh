python feature_extractor.py --dataset cifar10 --device cuda:0 --epochs 20
python generate_test_bags.py cifar10
zip -r cifar10_testbags.zip cifar10_testbags