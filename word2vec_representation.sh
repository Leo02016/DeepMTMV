# the script to generate word2vec representation
gcc -o word2vec word2vec.c
./word2vec -train ./data/web_content.txt -output ./data/web_content_representation.txt -cbow 1 -size 30 -window 8 -negative 25 -min-count 3 -binary 0
./word2vec -train ./data/web_links.txt -output ./data/web_links_representation.txt -cbow 1 -size 30 -window 8 -negative 25 -min-count 1 -binary 0
./word2vec -train ./data/web_title.txt -output ./data/web_title_representation.txt -cbow 1 -size 30 -window 8 -negative 25 -min-count 1 -binary 0
