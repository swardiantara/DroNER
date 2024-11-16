rm -r cache_dir
python fine_tune.py --model_type=electra

rm -r cache_dir
python fine_tune.py --model_type=bert

rm -r cache_dir
python fine_tune.py --model_type=roberta

rm -r cache_dir
python fine_tune.py --model_type=distilbert

rm -r cache_dir
python fine_tune.py --model_type=distilroberta

rm -r cache_dir
python fine_tune.py --model_type=xlnet