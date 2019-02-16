# Download annotations
wget http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip
unzip dataset.zip
rm dataset.zip
rm -r __MACOSX
mv dataset/* matlab_annos/
rm -r dataset

# Download images
wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
unzip sg_dataset.zip
rm sg_dataset.zip
mv sg_dataset/sg_test_images/* sg_dataset/sg_train_images/
rm -r sg_dataset/sg_test_images/
rm sg_dataset/sg_test_annotations.json
rm sg_dataset/sg_train_annotations.json
mv sg_dataset/sg_train_images/ sg_dataset/images/

echo Check config.yaml and run prepare_data.py!
