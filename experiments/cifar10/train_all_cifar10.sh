python train_cifar10.py --train_name cifar10_histnet_final --dataset cifar10 --network histnet --network_parameters ../parameters/histnet.json --feature_extraction rff --loss_function mrae --cuda_device cuda:0 &
python train_cifar10.py --train_name cifar10_gmnet_final --dataset cifar10 --network gmnet --network_parameters ../parameters/gmnet.json --feature_extraction nofe --loss_function mrae --cuda_device cuda:1 &
wait
python train_cifar10.py --train_name cifar10_gmnet_01_final --dataset cifar10 --network gmnet --network_parameters ../parameters/gmnet_reg01.json --feature_extraction nofe --loss_function mrae --cuda_device cuda:0 &
python train_cifar10.py --train_name cifar10_gmnet_001_final --dataset cifar10 --network gmnet --network_parameters ../parameters/gmnet_reg001.json --feature_extraction nofe --loss_function mrae --cuda_device cuda:1 &
wait
python train_cifar10.py --train_name cifar10_gmnet_0001_final --dataset cifar10 --network gmnet --network_parameters ../parameters/gmnet_reg0001.json --feature_extraction nofe --loss_function mrae --cuda_device cuda:0 &
python train_cifar10.py --train_name cifar10_deepsets_median_final --dataset cifar10 --network deepsets --network_parameters ../parameters/dqn_median.json --feature_extraction rff --loss_function mrae --cuda_device cuda:1 &
wait
python train_cifar10.py --train_name cifar10_deepsets_avg_final --dataset cifar10 --network deepsets --network_parameters ../parameters/dqn_avg.json --feature_extraction rff --loss_function mrae --cuda_device cuda:0 &
python train_cifar10.py --train_name cifar10_deepsets_max_final --dataset cifar10 --network deepsets --network_parameters ../parameters/dqn_max.json --feature_extraction rff --loss_function mrae --cuda_device cuda:1 &