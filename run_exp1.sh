
python main_IW.py --exp_name test_w-aux_0 --lam 10 --seed 42
python main_IW.py --exp_name test_w-aux_1 --lam 10 --seed 0
python main_IW.py --exp_name test_w-aux_2 --lam 10 --seed 105
python main_IW.py --exp_name test_w-aux_3 --lam 10 --seed 7
python main_IW.py --exp_name test_w-aux_4 --lam 100 --seed 59

python IW_ablation.py --exp_name vanilla-update_w-aux_0 --lam 10 --seed 42
python IW_ablation.py --exp_name vanilla-update_w-aux_1 --lam 10 --seed 0
python IW_ablation.py --exp_name vanilla-update_w-aux_2 --lam 10 --seed 105
python IW_ablation.py --exp_name vanilla-update_w-aux_3 --lam 10 --seed 37
python IW_ablation.py --exp_name vanilla-update_w-aux_4 --lam 10 --seed 49

#python main_IW.py --exp_name clip_no-update --lam 0 --seed 42 --update_episode -1


#python main_IW.py --exp_name nn-w-proj_0 --lam 0 --seed 42
#python main_IW.py --exp_name nn-w-proj_1 --lam 0 --seed 0
#python main_IW.py --exp_name nn-w-proj_2 --lam 0 --seed 15
#python main_IW.py --exp_name nn-w-proj_3 --lam 0 --seed 37
#python main_IW.py --exp_name nn-w-proj_4 --lam 0 --seed 49

#python IW_ablation.py --exp_name _0 --lam 0 --seed 42
#python IW_ablation.py --exp_name vannilla-update_2 --lam 0 --seed 0
#python IW_ablation.py --exp_name vannilla-update_3 --lam 0 --seed 15
#python IW_ablation.py --exp_name vannilla-update_4 --lam 0 --seed 37
#python IW_ablation.py --exp_name vannilla-update_5 --lam 0 --seed 49
