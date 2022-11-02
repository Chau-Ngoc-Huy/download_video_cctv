activate:
	source env/bin/activate

install: 
	pip install -r requirements.txt

run:
	python3 main.py

clean_image:
	rm ./data/cam_1/*.png
	rm ./data/cam_2/*.png