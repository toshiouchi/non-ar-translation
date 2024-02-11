# -*- coding: utf-8 -*-

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import math

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

import json

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# 認識エラー率を計算するモジュールをインポート
import levenshtein

# モデルの定義をインポート
from my_model import MyE2EModel

# json形式の入出力を行うモジュールをインポート
import json

# os, sys, shutilモジュールをインポート
import os
import sys
import shutil

def ctc_simple_decode(int_vector, token_list):
    ''' 以下の手順で，フレーム単位のCTC出力をトークン列に変換する
        1. 同じ文字が連続して出現する場合は削除
        2. blank を削除
    int_vector: フレーム単位のCTC出力(整数値列)
    token_list: トークンリスト
    output:     トークン列
    '''
    #print( "en3:{}".format( len( token_list ) ) )
    #print( "en4:{}".format( token_list[9656] ) )
    #print( "en5:{}".format( token_list[9657] ) )
    # 出力文字列
    output = []
    # 一つ前フレームの文字番号
    prev_token = -1
    # フレーム毎の出力文字系列を前から順番にチェックしていく
    for n in int_vector:
        #print( "n:{}".format( n ) )
        #print( " n:{}".format( n ))
        #print( " prev_token:{}".format( prev_token ))
        if n != prev_token:
            # 1. 前フレームと同じトークンではない
            if n != 0:
                # 2. かつ，blank(番号=0)ではない
                # --> token_listから対応する文字を抽出し，
                #     出力文字列に加える
                #print( "n:{}".format( n ) )
                #output.append(token_list[str(n)])
                if token_list[str(n)] != "<sos>" and token_list[str(n)] != "<eos>" and token_list[str(n)] != '.':
                    output.append( " " + token_list[str(n)])
                else:
                    output.append( token_list[str(n)])
                if token_list[str(n)] == '<eos>':
                    break
            # 前フレームのトークンを更新
            prev_token = n
    return output


#
# メイン関数
#
if __name__ == "__main__":
    
    #
    # 設定ここから
    #

    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'char'

    # 学習データの特徴量(feats.scp)が存在するディレクトリ
    feat_dir_train = '../01compute_features/fbank/train_large'
    # 開発データの特徴量(Feats.scp)が存在するディレクトリ
    feat_dir_dev = '../01compute_features/fbank/dev'

    # 実験ディレクトリ
    # train_set_name = 'train_small' or 'train_large'
    train_set_name = os.path.basename(feat_dir_train) 
    exp_dir = './exp_' + os.path.basename(feat_dir_train) 

    # 学習/開発データの特徴量リストファイル
    feat_scp_train = os.path.join(feat_dir_train, 'feats.scp')
    feat_scp_dev = os.path.join(feat_dir_dev, 'feats.scp')

    # 学習/開発データのラベルファイル
    label_train = os.path.join(exp_dir, 'data', unit,
                               'label_'+train_set_name)
    label_dev = os.path.join(exp_dir, 'data', unit,
                             'label_dev')
    
    # 訓練データから計算された特徴量の平均/標準偏差ファイル
    mean_std_file = os.path.join(feat_dir_train, 'mean_std.txt')

    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', unit,
                                   'token_list')
    # 学習結果を出力するディレクトリ
    output_dir = os.path.join(exp_dir, unit+'_model_non_ar_conv_005')

    # ミニバッチに含める発話数
    #batch_size = 10
    #batch_size = 128
    batch_size = 16

    # 最大エポック数
    #max_num_epoch = 60
    #max_num_epoch = 200
    #max_num_epoch = 25
    max_num_epoch = 15
    #max_num_epoch = 2

    # Encoderの設定
    enc_vocab = 61655
    # レイヤー数
    enc_num_layers = 12
    # encoder の head の数
    enc_num_heads = 8
    # Encoder の Attention block の次元数
    enc_att_hidden_dim = 1024
    # encoder 入力の時間の最大数
    enc_input_maxlen = 300
    # Encoder の Attention Bolock の kernel_size
    enc_att_kernel_size = [5,1]
    # Encoder の Attention Block の filter_size
    enc_att_filter_size = 4096
    # Encoder の dropout
    enc_dropout = 0.1
    
    #アップサンプリングの割合
    up_rate = 3.0

    # Decoderの設定
    dec_vocab = 57854
    # attnesion blockのレイヤー数
    dec_num_layers = 12
    # decoder の head の数
    dec_num_heads = 8
    # Decoder の Attention block の次元数
    dec_att_hidden_dim = 1024
    # decoder 入力( decoder targets, encoder_outs ではない）の時間の最大数
    dec_target_maxlen = 900
    # Deccoder の Attention Bolock の kernel_size
    dec_att_kernel_size = [5,1]
    # Decoder の Attention Block の filter_size
    dec_att_filter_size = 4096
    # Decoder の dropout
    dec_dropout = 0.1

    # 初期学習率
    #initial_learning_rate = 1.0
    #initial_learning_rate = 0.1
    initial_learning_rate = 0.00001

    # Clipping Gradientの閾値
    #clip_grad_threshold = 5.0
    clip_grad_threshold = 1.0

    # 学習率の減衰やEarly stoppingの
    # 判定を開始するエポック数
    # (= 最低限このエポックまではどれだけ
    # validation結果が悪くても学習を続ける)
    lr_decay_start_epoch = 7

    # 学習率を減衰する割合
    # (減衰後学習率 <- 現在の学習率*lr_decay_factor)
    # 1.0以上なら，減衰させない
    lr_decay_factor = 0.5

    # Early stoppingの閾値
    # 最低損失値を更新しない場合が
    # 何エポック続けば学習を打ち切るか
    early_stop_threshold = 3

    # 学習過程で，認識エラー率を計算するか否か
    # 認識エラー率の計算は時間がかかるので注意
    # (ここではvalidationフェーズのみTrue(計算する)にしている)
    evaluate_error = {'train': True, 'validation': True}

    #
    # 設定ここまで
    #
    
    # Attention重み行列情報の保存先
    out_att_dir = os.path.join(output_dir, 'att_matrix')
    
    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_att_dir, exist_ok=True)

    # 設定を辞書形式にする
    config = {'enc_vocab': enc_vocab,
              'enc_num_layers': enc_num_layers,
              'enc_num_heads': enc_num_heads,
              'enc_input_maxlen' : enc_input_maxlen,
              'enc_att_hidden_dim': enc_att_hidden_dim,
              'enc_att_kernel_size': enc_att_kernel_size,
              'enc_att_filter_size': enc_att_filter_size,
              'up_rate': up_rate,
              'enc_dropout_rate': enc_dropout,
              'dec_vocab': dec_vocab,
              'dec_num_layers': dec_num_layers,
              'dec_num_heads': dec_num_heads,
              'dec_target_maxlen': dec_target_maxlen,
              'dec_att_hidden_dim': dec_att_hidden_dim,
              'dec_att_kernel_size': dec_att_kernel_size,
              'dec_att_filter_size': dec_att_filter_size,
              'dec_dropout_rate': dec_dropout,
              'batch_size': batch_size,
              'max_num_epoch': max_num_epoch,
              'clip_grad_threshold': clip_grad_threshold,
              'initial_learning_rate': initial_learning_rate,
              'lr_decay_start_epoch': lr_decay_start_epoch, 
              'lr_decay_factor': lr_decay_factor,
              'early_stop_threshold': early_stop_threshold
             }

    # 設定をJSON形式で保存する
    conf_file = os.path.join(output_dir, 'config.json')
    with open(conf_file, mode='w', encoding='utf-8' ) as f:
        json.dump(config, f, indent=4)

    ## 特徴量の平均/標準偏差ファイルを読み込む
    #with open(mean_std_file, mode='r', encoding='utf-8' ) as f:
    #    # 全行読み込み
    #    lines = f.readlines()
    #    # 1行目(0始まり)が平均値ベクトル(mean)，
    #    # 3行目が標準偏差ベクトル(std)
    #    mean_line = lines[1]
    #    std_line = lines[3]
    #    # スペース区切りのリストに変換
    #    feat_mean = mean_line.split()
    #    feat_std = std_line.split()
    #    # numpy arrayに変換
    #    feat_mean = np.array(feat_mean, 
    #                            dtype=np.float32)
    #    feat_std = np.array(feat_std, 
    #                           dtype=np.float32)
    ## 平均/標準偏差ファイルをコピーする
    #shutil.copyfile(mean_std_file,
    #                os.path.join(output_dir, 'mean_std.txt'))

    ## 次元数の情報を得る
    #feat_dim = np.size(feat_mean)

    # トークンリストをdictionary型で読み込む
    # このとき，0番目は blank と定義する
    # (ただし，このプログラムではblankは使われない)
    token_list = {0: '<blank>'}
    token_list_en = {0: '<blank>'}
    # token_list, token_list_en, pad_id = 4
    pad_id = 4
    token_list_path = "idx_to_wakati.json"
    with open( token_list_path, mode="r" , encoding="utf-8" ) as f:
        token_list = json.load( f )
    f.close()
    token_list_en_path = "idx_to_word.json"
    with open( token_list_en_path, mode="r" , encoding="utf-8" ) as f:
        token_list_en = json.load( f )
    f.close()
    #token_list_en[20175] = '<blank>'
    #with open(token_list_path, mode='r', encoding='utf-8' ) as f:
    #    # 1行ずつ読み込む
    #    for line in f: 
    #        # 読み込んだ行をスペースで区切り，
    #        # リスト型の変数にする
    #        parts = line.split()
    #        # 0番目の要素がトークン，1番目の要素がID
    #        token_list[int(parts[1])] = parts[0]

    # <eos>トークンをユニットリストの末尾に追加
    #eos_id = len(token_list)
    #token_list[eos_id] = '<eos>'
    # 本プログラムでは、<sos>と<eos>を
    # 同じトークンとして扱う
    #sos_id = eos_id
    #sos_id = len(token_list)
    #token_list[sos_id] = '<sos>'

    # トークン数(blankを含む)
    num_tokens = len(token_list)
    #print( "jp:{}".format( num_tokens ) )
    num_tokens_en = len(token_list_en)
    #print( "en:{}".format( num_tokens_en ) )
    
    # ニューラルネットワークモデルを作成する
    # 入力の次元数は特徴量の次元数，
    # 出力の次元数はトークン数となる
    model = MyE2EModel(enc_vocab = enc_vocab,
                       enc_num_layers = enc_num_layers,
                       enc_att_hidden_dim=enc_att_hidden_dim,
                       enc_num_heads = enc_num_heads,
                       enc_input_maxlen = enc_input_maxlen, 
                       enc_att_kernel_size=enc_att_kernel_size,
                       enc_att_filter_size=enc_att_filter_size,
                       enc_dropout_rate = enc_dropout,
                       up_rate = up_rate,
                       dec_vocab = dec_vocab,
                       dec_num_layers = dec_num_layers,
                       dec_att_hidden_dim=dec_att_hidden_dim,
                       dec_num_heads = dec_num_heads, 
                       dec_target_maxlen = dec_target_maxlen,
                       dec_att_kernel_size = dec_att_kernel_size,
                       dec_att_filter_size = dec_att_filter_size,
                       dec_dropout_rate = dec_dropout,
                       )
    print(model)

    # オプティマイザを定義
    #optimizer = optim.Adadelta(model.parameters(),
    #                           lr=initial_learning_rate,
    #                           rho=0.95,
    #                           eps=1e-8,
    #                           weight_decay=0.0)
    optimizer = optim.Adam(model.parameters(),
                                lr=initial_learning_rate)

    # 訓練/開発データのデータセットを作成する
    #train_dataset = SequenceDataset(feat_scp_train,
    #                                label_train,
    #                                feat_mean,
    #                                feat_std)

    # 開発データのデータセットを作成する
    #dev_dataset = SequenceDataset(feat_scp_dev,
    #                              label_dev,
    #                              feat_mean,
    #                              feat_std)

    train_dataset = SequenceDataset( "ids_train.txt", pad_id )
    dev_dataset = SequenceDataset( "ids_dev.txt", pad_id )
    #train_dataset = SequenceDataset( "ids_test1.txt", pad_id )
    #dev_dataset = SequenceDataset( "ids_test1.txt", pad_id )
    
    # 訓練データのDataLoaderを呼び出す
    # 訓練データはシャッフルして用いる
    #  (num_workerは大きい程処理が速くなりますが，
    #   PCに負担が出ます．PCのスペックに応じて
    #   設定してください)
    train_loader = DataLoader(train_dataset,
    #train_loader = DataLoader(dev_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    # 開発データのDataLoaderを呼び出す
    # 開発データはデータはシャッフルしない
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    # クロスエントロピー損失を用いる．ゼロ埋めしているラベルを
    # 損失計算に考慮しないようにするため，ignore_index=0を設定
    #criterion = nn.CrossEntropyLoss(ignore_index=0)
    # CTC損失関数を呼び出す．
    # blankは0番目と定義する．
    criterion = nn.CTCLoss(blank=0, reduction='mean',zero_infinity=False)    
    

    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # モデルをトレーニングモードに設定する
    model.train()

    # 訓練データの処理と開発データの処理を
    # for でシンプルに記述するために，辞書データ化しておく
    dataset_loader = {'train': train_loader,
    #dataset_loader = {'train': dev_loader,
                      'validation': dev_loader}

    # 各エポックにおける損失値と誤り率の履歴
    loss_history = {'train': [],
                    'validation': []}
    error_history = {'train': [],
                     'validation': []}

    # 本プログラムでは，validation時の損失値が
    # 最も低かったモデルを保存する．
    # そのため，最も低い損失値，
    # そのときのモデルとエポック数を記憶しておく
    best_loss = -1
    best_model = None
    best_epoch = 0
    # Early stoppingフラグ．Trueになると学習を打ち切る
    early_stop_flag = False
    # Early stopping判定用(損失値の最低値が
    # 更新されないエポックが何回続いているか)のカウンタ
    counter_for_early_stop = 0

    # ログファイルの準備
    log_file = open(os.path.join(output_dir,
                                 'log.txt'),
                                 mode='w', encoding='utf-8' )
    log2_file = open( 'out_005.log', mode='w', encoding='utf-8' )
    log_file.write('epoch\ttrain loss\t'\
                   'train err\tvalid loss\tvalid err')

    # エポックの数だけループ
    for epoch in range(max_num_epoch):
        # early stopフラグが立っている場合は，
        # 学習を打ち切る
        if early_stop_flag:
            print('    Early stopping.'\
                  ' (early_stop_threshold = %d)' \
                  % (early_stop_threshold))
            log_file.write('\n    Early stopping.'\
                           ' (early_stop_threshold = %d)' \
                           % (early_stop_threshold))
            log2_file.write('\n    Early stopping.'\
                           ' (early_stop_threshold = %d)' \
                           % (early_stop_threshold))
            break

        # エポック数を表示
        print('epoch %d/%d:' % (epoch+1, max_num_epoch))
        log_file.write('\n%d\t' % (epoch+1))
        log2_file.write('epoch %d/%d:\n' % (epoch+1, max_num_epoch))

        # trainフェーズとvalidationフェーズを交互に実施する
        for phase in ['train', 'validation']:
            # このエポックにおける累積損失値と発話数
            total_loss = 0
            total_utt = 0
            # このエポックにおける累積認識誤り文字数と総文字数
            total_error = 0
            total_token_length = 0

            # 各フェーズのDataLoaderから1ミニバッチ
            # ずつ取り出して処理する．
            # これを全ミニバッチ処理が終わるまで繰り返す．
            # ミニバッチに含まれるデータは，
            # 音声特徴量，ラベル，フレーム数，
            # ラベル長，発話ID
            n_batch = 0
            for (features, labels, feat_lens,
                 label_lens) \
                    in dataset_loader[phase]:
                n_batch += 1
                ##
                ## ラベルの末尾に<eos>を付与する最初に<sos>
                ##
                ## ゼロ埋めにより全体の長さを1増やす
                #labels = torch.cat( ( torch.zeros(labels.size()[0],1,dtype=torch.long), labels), dim = 1 )
                #labels = torch.cat((labels,
                #                    torch.zeros(labels.size()[0],
                #                    1,
                #                    dtype=torch.long)), dim=1)
                ## 末尾に<eos>追加、最初に<sos> を追加
                #for m, length in enumerate(label_lens):
                #    labels[m][0] = sos_id
                #    labels[m][length+1] = eos_id
                #label_lens += 2

                # 現時点でラベルのテンソルサイズは
                # [発話数 x 全データの最大ラベル長]
                # これを[発話数 x バッチ内の最大ラベル長]
                # に切る。(decoder部の冗長な処理を少なくするため。)
                features = features[:,:torch.max(feat_lens)]
                labels = labels[:,:torch.max(label_lens)]

                # CUDAが使える場合はデータをGPUに，
                # そうでなければCPUに配置する
                features, labels = \
                    features.to(device), labels.to(device)

                # 勾配をリセット
                optimizer.zero_grad()

                # モデルの出力を計算(フォワード処理)
                #outputs, outputs_lens = model(features, feat_lens, dec_input, dec_input_lens )
                outputs, outputs_lens = model(features, feat_lens )
                outputs = F.log_softmax( outputs, dim=2 )
                
                #if outputs.size(1) <= int( labels.size(1) * 1.5 ):
                ##if outputs.size(1) <= labels.size(1):
                #     outputs2 = torch.ones( ( outputs.size(0), int( labels.size(1) * 1.5 ), outputs.size(2 ) ) ) * pad_id
                #     #outputs2 = torch.ones( ( outputs.size(0), labels.size(1) * 2, outputs.size(2 ) ) ) * pad_id
                #     outputs2[:,:outputs.size(1), :] = outputs
                #     outputs = outputs2
                #     #print( "gyakuten" )

                # クロスエントロピー損失関数の入力は
                # [(バッチ）ｘ（クラス）ｘ（時間）」なので、transpose する。
                # target は、hot_vector にしないで、「（バッチ）×（時間）」のままで良い。
                #print( " size of outputs.transpose(1,2):{}".format( outputs.transpose(1,2).size() ))
                #print( " size of dec_target:{}".format( dec_target.size() ) )
                #loss = criterion( outputs.transpose(1,2).to(device), dec_target.to(device) )
                # 損失値を計算する．このとき，CTCLossへの入力は
                # [フレーム数 x バッチサイズ x クラス数] 
                # である必要があるため，テンソルの0軸と1軸を
                # 転置(transpose(0,1))した上で入力する
                #T = outputs.size(1)
                #outputs_lens[outputs_lens > T] = T
                #for n in range( len( outputs_lens ) ):
                #    if outputs_lens[n] <= int( label_lens[n] * 1.5 ):
                #    #if outputs_lens[n] <= label_lens[n]:
                #        #outputs_lens[n] = label_lens[n] + 1
                #        outputs_lens[n] = int( label_lens[n] * 1.5 )
                #        #outputs_lens[n] = label_lens[n] * 2
                #out_lens = outputs_lens
                #print( "out_lens:{}".format( out_lens ) )
                #for n in outputs:
                #    for t in n:
                #        for c in t:
                #            if c == math.inf or c == math.nan:
                #                print( "c:{}".format( c ) )
                #if np.any( (outputs[:,:,:].cpu().detach().numpy() == np.inf) ) or np.any( (outputs[:,:,:].cpu().detach().numpy() == np.nan) ):
                #if torch.any(torch.isnan( outputs )) or torch.any(torch.isinf( outputs )):
                #    print( "nan or inf" )
                #print( "outputs:{}".format( outputs ))
                #print( "labels:{}".format( labels ))
                #if torch.any( out_lens < 5 ):
                #    print( "out_lens:{}".format( out_lens ))
                #print( "label_lens:{}".format( label_lens ) )
                #loss = criterion(outputs.transpose(0, 1),labels,out_lens,label_lens)
                loss = criterion(outputs.transpose(0, 1),labels,outputs_lens,label_lens)
                
                #loss *= np.mean(label_lens.numpy())

                # 訓練フェーズの場合は，誤差逆伝搬を実行し，
                # モデルパラメータを更新する
                if phase == 'train':
                    # 勾配を計算する
                    loss.backward()
                    # Cliping Gradient により勾配が
                    # 閾値以下になるよう調整する
                    torch.nn.utils.clip_grad_norm_(\
                                              model.parameters(),
                                              clip_grad_threshold)
                    # オプティマイザにより，パラメータを更新する
                    optimizer.step()

                # 認識エラーの算出をTrueにしている場合は，算出する
                if evaluate_error[phase]:
                    # バッチ内の1発話ごとに誤りを計算
                    jpn_st = []
                    reference_st = []
                    hypothesis_st = []
                    for n in range(outputs.size(0)):
                        # 各ステップのデコーダ出力を得る
                        _, hyp_per_frame = torch.max(outputs[n], 1)
                        # numpy.array型に変換
                        hyp_per_frame = hyp_per_frame.cpu().numpy()
                        # 認識結果の文字列を取得
                        hypothesis = \
                           ctc_simple_decode(hyp_per_frame,
                                      token_list_en)
                        # 正解の文字列を取得
                        reference = []
                        for m in labels[n][:label_lens[n]].cpu().numpy().astype( np.int32 ):
                            if token_list_en[str(m)] != "<sos>" and token_list_en[str(m)] != "<eos>" and token_list_en[str(m)] != '.':
                                reference.append( " " + token_list_en[str(m)])
                            else:
                                reference.append( token_list_en[str(m)])
                            
                            
                        # 正解の文字列を取得
                        jpn = []
                        for m in features[n][:feat_lens[n]].cpu().numpy().astype( np.int32 ):
                            #if token_list_en[m] != "<sos>" and token_list_en[m] != "<eos>":
                            jpn.append(token_list[str(m)].replace( "\n", ""))

                        # 認識誤りを計算
                        (error, substitute, 
                         delete, insert, ref_length) = \
                            levenshtein.calculate_error(hypothesis,
                                                        reference)
                        # 誤り文字数を累積する
                        total_error += error
                        # 文字の総数を累積する
                        total_token_length += ref_length
                        if n < 4 and n_batch == len( dataset_loader[phase] ):
                            sentence1 = "{:>9s}, japanese  :{}".format( phase, ''.join(jpn) )
                            jpn_st.append( sentence1  )
                            sentence1 = "{:>9s}, reference :{}".format( phase, ''.join(reference) )
                            reference_st.append( sentence1 )
                            sentence1 = "{:>9s}, hypothesis:{}".format( phase, ''.join(hypothesis) )
                            hypothesis_st.append( sentence1 )
                            #print( "jpn_st:".format( jpn_st[n] ) )
                            #print( "reference_st:".format( reference_st[n] ) )
                            #print( "hypothesis_st:".format( hypothesis_st[n] ) )
                            #print( "%12s, japanese  :%s" % (phase,''.join(jpn) ) )
                            #print( "%12s, reference :%s" % (phase,''.join(reference) ) )
                            #print( "%12s, hypothesis:%s" % (phase,''.join(hypothesis) ) )
                            #log2_file.wirte( "%12s, japanese  :%s\n" % (phase,''.join(jpn) ) )
                            #log2_file.wirte( "%12s, reference :%s\n" % (phase,''.join(reference) ) )
                            #log2_file.wirte( "%12s, hypothesis:%s\n" % (phase,''.join(hypothesis) ) )


                # 損失値を累積する
                total_loss += loss.item()
                # 処理した発話数をカウントする
                total_utt += outputs.size(0)

                if n_batch % 5 == 0:
                    print( "n_batch:{},phase:{:>9s},avg_loss:{:>.3e}, avg_error_rate:{:>.3e}".format( n_batch, phase, total_loss / total_utt, total_error * 100.0 / total_token_length) )
                    log2_file.write( "n_batch:{},phase:{:>9s},avg_loss:{:>.3e}, avg_error_rate:{:>.3e}\n".format( n_batch, phase, total_loss / total_utt, total_error * 100.0 / total_token_length) )
                    log2_file.flush()

                #print( "n_batch:{}".format( n_batch ) )
                #print( "len( dataset_loader[phase] ):{}".format( len( dataset_loader[phase] ) ) )
                if n_batch == len( dataset_loader[phase] ):
                    for n in range( len( jpn_st ) ):
                        #print( "n:",n )
                        print( "{}".format(jpn_st[n] ))
                        print( "{}".format(reference_st[n] ))
                        print( "{}".format(hypothesis_st[n] ))
                        log2_file.write( "{}\n".format(jpn_st[n] ))
                        log2_file.write( "{}\n".format(reference_st[n] ))
                        log2_file.write( "{}\n".format(hypothesis_st[n] ))

                        

            #
            # このフェーズにおいて，1エポック終了
            # 損失値，認識エラー率，モデルの保存等を行う
            # 

            # 損失値の累積値を，処理した発話数で割る
            epoch_loss = total_loss / total_utt
            # 画面とログファイルに出力する
            #print("n_batch:{}".format( n_batch ))
            print('    %s loss: %f' \
                  % (phase, epoch_loss))
            log_file.write('%.6f\t' % (epoch_loss))
            log2_file.write('    %s loss: %f\n' \
                  % (phase, epoch_loss))
            # 履歴に加える
            loss_history[phase].append(epoch_loss)

            # 認識エラー率を計算する
            if evaluate_error[phase]:
                # 総誤りトークン数を，
                # 総トークン数で割ってエラー率に換算
                epoch_error = 100.0 * total_error \
                            / total_token_length
                # 画面とログファイルに出力する
                print('    %s token error rate: %f %%' \
                    % (phase, epoch_error))
                log2_file.write('    %s token error rate: %f %%\n' \
                    % (phase, epoch_error))
                log_file.write('%.6f\t' % (epoch_error))
                # 履歴に加える
                error_history[phase].append(epoch_error)
            else:
                # エラー率を計算していない場合
                log_file.write('     ---     \t')

            #
            # validationフェーズ特有の処理
            #
            #if phase == 'validation':
            if phase == 'train':
                if epoch == 0 or best_loss > epoch_loss:
                    # 損失値が最低値を更新した場合は，
                    # その時のモデルを保存する
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 
                               output_dir+'/best_model.pt')
                    best_epoch = epoch
                    # Early stopping判定用の
                    # カウンタをリセットする
                    counter_for_early_stop = 0
                else:
                    # 最低値を更新しておらず，
                    if epoch+1 >= lr_decay_start_epoch:
                        # かつlr_decay_start_epoch以上の
                        # エポックに達している場合
                        if counter_for_early_stop+1 \
                               >= early_stop_threshold:
                            # 更新していないエポックが，
                            # 閾値回数以上続いている場合，
                            # Early stopping フラグを立てる
                            early_stop_flag = True
                        else:
                            # Early stopping条件に
                            # 達していない場合は
                            # 学習率を減衰させて学習続行
                            if lr_decay_factor < 1.0:
                                for i, param_group \
                                      in enumerate(\
                                      optimizer.param_groups):
                                    if i == 0:
                                        lr = param_group['lr']
                                        dlr = lr_decay_factor \
                                            * lr
                                        print('    (Decay '\
                                          'learning rate:'\
                                          ' %f -> %f)' \
                                          % (lr, dlr))
                                        log_file.write(\
                                          '(Decay learning'\
                                          ' rate: %f -> %f)'\
                                           % (lr, dlr))
                                        log2_file.write(\
                                          '(Decay learning'\
                                          ' rate: %f -> %f)'\
                                           % (lr, dlr))
                                    param_group['lr'] = dlr
                            # Early stopping判定用の
                            # カウンタを増やす
                            counter_for_early_stop += 1

    #
    # 全エポック終了
    # 学習済みモデルの保存とログの書き込みを行う
    #
    print('---------------Summary'\
          '------------------')
    log_file.write('\n---------------Summary'\
                   '------------------\n')
    log2_file.write('\n---------------Summary'\
                   '------------------\n')

    # 最終エポックのモデルを保存する
    torch.save(model.state_dict(), 
               os.path.join(output_dir,'final_model.pt'))
    print('Final epoch model -> %s/final_model.pt' \
          % (output_dir))
    log_file.write('Final epoch model ->'\
                   ' %s/final_model.pt\n' \
                   % (output_dir))
    log2_file.write('Final epoch model ->'\
                   ' %s/final_model.pt\n' \
                   % (output_dir))


    # 最終エポックの情報
    for phase in ['train', 'validation']:
        # 最終エポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][-1]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][-1]))
        log2_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][-1]))
        # 最終エポックのエラー率を出力    
        if evaluate_error[phase]:
            print('    %s token error rate: %f %%' \
                % (phase, error_history[phase][-1]))
            log_file.write('    %s token error rate: %f %%\n' \
                % (phase, error_history[phase][-1]))
            log2_file.write('    %s token error rate: %f %%\n' \
                % (phase, error_history[phase][-1]))
        else:
            print('    %s token error rate: (not evaluated)' \
                % (phase))
            log_file.write('    %s token error rate: '\
                '(not evaluated)\n' % (phase))
            log2_file.write('    %s token error rate: '\
                '(not evaluated)\n' % (phase))


    # ベストエポックの情報
    # (validationの損失が最小だったエポック)
    print('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt' \
          % (best_epoch+1, output_dir))
    log_file.write('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt\n' \
          % (best_epoch+1, output_dir))
    log2_file.write('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt\n' \
          % (best_epoch+1, output_dir))
    for phase in ['train', 'validation']:
        # ベストエポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][best_epoch]))
        log_file.write('    %s loss: %f\n' \
              % (phase, loss_history[phase][best_epoch]))
        log2_file.write('    %s loss: %f\n' \
              % (phase, loss_history[phase][best_epoch]))
        # ベストエポックのエラー率を出力
        if evaluate_error[phase]:
            print('    %s token error rate: %f %%' \
                  % (phase, error_history[phase][best_epoch]))
            log_file.write('    %s token error rate: %f %%\n' \
                  % (phase, error_history[phase][best_epoch]))
            log2_file.write('    %s token error rate: %f %%\n' \
                  % (phase, error_history[phase][best_epoch]))
        else:
            print('    %s token error rate: '\
                  '(not evaluated)' % (phase))
            log_file.write('    %s token error rate: '\
                  '(not evaluated)\n' % (phase))
            log2_file.write('    %s token error rate: '\
                  '(not evaluated)\n' % (phase))

    # 損失値の履歴(Learning Curve)グラフにして保存する
    fig1 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(loss_history[phase],
                 label=phase+' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig1.legend()
    fig1.savefig(output_dir+'/loss.png')

    # 認識誤り率の履歴グラフにして保存する
    fig2 = plt.figure()
    for phase in ['train', 'validation']:
        if evaluate_error[phase]:
            plt.plot(error_history[phase],
                     label=phase+' error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [%]')
    fig2.legend()
    fig2.savefig(output_dir+'/error.png')

    # ログファイルを閉じる
    log_file.close()
    log2_file.close()


