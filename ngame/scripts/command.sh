python runner.py --pipeline NGAME --work_dir /home/scai/phd/aiz218323/Projects/XC --version 0 --seed 100 --config configs/NGAME/LF-AmazonTitles-1.3M.json
python runner.py --pipeline NGAME --work_dir /home/scai/phd/aiz218323/Projects/XC --version 1 --seed 100 --config configs/NGAME/LF-AmazonTitles-1.3M_meta-category.json

python runner.py --pipeline NGAME --work_dir /home/scai/phd/aiz218323/Projects/XC --version 0 --seed 100 --config configs/NGAME/LF-WikiSeeAlsoTitles-320K.json
python runner.py --pipeline NGAME --work_dir /home/scai/phd/aiz218323/Projects/XC --version 1 --seed 100 --config configs/NGAME/LF-WikiSeeAlsoTitles-320K_meta-hyperlink.json

python runner.py --pipeline NGAME --work_dir /home/scai/phd/aiz218323/Projects/XC --version 0 --seed 100 --config configs/NGAME/LF-WikiTitles-500K.json
python runner.py --pipeline NGAME --work_dir /home/scai/phd/aiz218323/Projects/XC --version 1 --seed 100 --config configs/NGAME/LF-WikiTitles-500K_meta-hyperlink.json
