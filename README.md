Video Captioning baseline implementation using Pytorch

Steps
1. preprocess videos and labels
   python prepro_feats.py --output_dir data/feats/resnet152 --model resnet152 --n_frame_steps 40  --gpu 4,5
   python prepro_vocab.py
   
2. Training a model
   python train.py --gpu 0 --epochs 3001 --batch_size 300 --checkpoint_path data/save --feats_dir data/feats/resnet152 --model S2VTAttModel  --with_c3d 1 --c3d_feats_dir data/feats/c3d_feats --dim_vid 4096
   
3. Test

   python eval.py --recover_opt data/save/opt_info.json --saved_model data/save/model_1000.pth --batch_size 100 --gpu 1
