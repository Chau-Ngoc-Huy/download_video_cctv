activate:
	source env/bin/activate

install: 
	pip install -r requirements.txt

run:
	python3.10 main.py

push_git:
	git add .
	git commit -m "test"
	git push

clean_image:
	rm ./data/cam_1/*.png
	rm ./data/cam_2/*.png