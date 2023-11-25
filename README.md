# non-ar-translation
non-autoregressive machine translation

Machine learning result

Japanese English machine translation was learned with 200,000 data. For 5,000 test data, WER was 37%.

Feature of program

Transformer with convolutional position wise feed forward network is used. 

Encoder has 12 layers, 8 heads, hidden dim 1024. Also decoder has 12 layers, 8 heads, hidden dim 1024.

Position wise feed forward network in both TransformerEncoder and TransformerDecoder has two convolutional layers which have filters (1024, 4096) , (4096, 1024), kernel sizes are 5 and 1, strides are both 1, layer norm and dropout with rate 0.1.

Input of encoder is Japanese indexes, it is embed, positional embedding is done. Then, the sum of embed values and positional embedding values are input in TransformerEncoder with self attention module.

Especially encoder output stretches to double with time axis using upsampling module. And, also input_lens which expresses input lengths of encoder is doubled and becomes the input of CTCLoss as output_lens.

TransformerDecoder is used as cross attention module. Source input is encoder ouput and target input is upsampled encoder output with positional embedding. 

Decoder(TransformerDecoder) output is input in a linear projection layer.

CTCLoss is used by loss calculation. 

loss = nn.CTCLoss(blank=0, reduction='mean',zero_infinity=False)( outputs.transpose(0,1), labels, output_lengths, labels_lens )

where outputs and output_lens are calculated with machine learning model. Labels and label_lens are teacher data.

CTCLoss outputs inf when output_lengths is not greater than labels_lens to some extent. So, if outputs is shorter than 1.5 times of labels with time axis, outputs is padded with pad_id to be 1.5 times the size of labels with time axis. And output_lens is also corrected.

CTCLoss is used, so ctc_simple_decode(int_vector, token_list) function in source code is used in order to decode outputs of model.inference.

Explanation of detail machine learning in Japanese
https://qiita.com/toshiouchi/items/f6932bb5c3fe1c04f489

