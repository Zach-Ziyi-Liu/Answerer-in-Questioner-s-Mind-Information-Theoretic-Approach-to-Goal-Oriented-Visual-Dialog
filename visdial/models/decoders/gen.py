import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from utils import utilities as utils


class Decoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 startToken,
                 endToken,
                 dropout=0,
                 **kwargs):
        super(Decoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers
        self.startToken = startToken
        self.endToken = endToken
        self.dropout = dropout

#         self.max_length = 10
#         self.attn = nn.Linear(self.rnnHiddenSize * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.rnnHiddenSize * 2, self.rnnHiddenSize)
        
        # Modules
        self.rnn = nn.LSTM(
            self.embedSize,
            self.rnnHiddenSize,
            self.numLayers,
            batch_first=True,
            dropout=self.dropout)
        self.outNet = nn.Linear(self.rnnHiddenSize, self.vocabSize)
        self.projection = nn.Sequential(
                        nn.Linear(self.rnnHiddenSize, 64),
                        nn.ReLU(True),
                        nn.Linear(64, 1)
                    )
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.dropout_p = 0.1
        self.dropout0 = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.rnnHiddenSize, self.rnnHiddenSize)
        self.concat = nn.Linear(self.rnnHiddenSize * 2, self.rnnHiddenSize)
       
    
    def my_log_softmax(self,x):
    
            size = x.size()
            res = F.log_softmax(x.squeeze())
            res = res.view(size[0], size[1], -1)
            return res
        
    def forward(self, encStates, inputSeq, enc_outputs = None, Attention = False, Self = False):
        '''
        Given encoder states, forward pass an input sequence 'inputSeq' to
        compute its log likelihood under the current decoder RNN state.

        Arguments:
            encStates: (H, C) Tuple of hidden and cell encoder states
            inputSeq: Input sequence for computing log probabilities

        Output:
            A (batchSize, length, vocabSize) sized tensor of log-probabilities
            obtained from feeding 'inputSeq' to decoder RNN at every time step

        Note:
            Maximizing the NLL of an input sequence involves feeding as input
            tokens from the GT (ground truth) sequence at every time step and
            maximizing the probability of the next token ("teacher forcing").
            See 'maskedNll' in utils/utilities.py where the log probability of
            the next time step token is indexed out for computing NLL loss.
        '''
        if inputSeq is not None:
            if Attention:
                       
                inputSeq = self.wordEmbed(inputSeq)
            
                inputSeq = self.dropout0(inputSeq)
                outputs, _ = self.rnn(inputSeq, encStates)
                outputs = F.dropout(outputs, self.dropout, training=self.training)
#                     print(outputs.shape,"\n")
                enc_outputs = self.attn(enc_outputs).unsqueeze(1).transpose(1,2)
#                     print(enc_outputs.shape,"\n")
                attn_energies = outputs.bmm(enc_outputs)
                attn_weights = self.my_log_softmax(attn_energies)
#                     print(attn_weights.shape,enc_outputs.transpose(1,2).shape)
                context = attn_weights.bmm(enc_outputs.transpose(1,2))
#                     context = attn_weights.bmm(outputs.tra nnspose(1,2))
#                     context = attn_weights.bmm(enc_outputs)
#                     print(context.shape)

                output_context = torch.cat((outputs, context), 2)

                output_context = self.concat(output_context)
#                     concat_output = F.relu(output_context)
#                     flatScores = self.outNet(flatScores)
#                     outputSize = outputs.size()
#                     flatOutputs = outputs.view(-1, outputSize[2])
                concat_output = F.tanh(output_context)
                outputSize = outputs.size()
                flatScores = self.outNet(concat_output)
                flatLogProbs = self.logSoftmax(flatScores)
                logProbs = flatLogProbs.view(outputSize[0], outputSize[1], -1)
                   

                   
                    
                  
        
        
        
#         class SelfAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim
      
#         )

#     def forward(self, encoder_outputs):
#         batch_size = encoder_outputs.size(0)
#         # (B, L, H) -> (B , L, 1)
#         energy = self.projection(encoder_outputs)
#         weights = F.softmax(energy.squeeze(-1), dim=1)
#         # (B, L, H) * (B, L, 1) -> (B, H)
#         outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
#         return outputs, weights

# class AttnClassifier(nn.Module):
#     def __init__(self, input_dim, embedding_dim, hidden_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim
#         self.hidden_dim = hidden_dim
#         self.embedding = nn.Embedding(input_dim, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
#         self.attention = SelfAttention(hidden_dim)
#         self.fc = nn.Linear(hidden_dim, 1)
        
#     def set_embedding(self, vectors):
#         self.embedding.weight.data.copy_(vectors)
        
#     def forward(self, inputs, lengths):
#         batch_size = inputs.size(1)
#         # (L, B)
#         embedded = self.embedding(inputs)
#         # (L, B, E)
#         packed_emb = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
#         out, hidden = self.lstm(packed_emb)
#         out = nn.utils.rnn.pad_packed_sequence(out)[0]
#         out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
#         # (L, B, H)
#         embedding, attn_weights = self.attention(out.transpose(0, 1))
#         # (B, HOP, H)
#         outputs = self.fc(embedding.view(batch_size, -1))
#         # (B, 1)
#         return outputs, attn_weights
    
    
#             elif Questioner = "self_attention":
#                     inputSeq = self.wordEmbed(inputSeq) 
#                     outputs, hidden = self.rnn(inputSeq, encStates)
#                     self.projection = nn.Sequential(
#                         nn.Linear(hidden_dim, 64),
#                         nn.ReLU(True),
#                         nn.Linear(64, 1)
#                     )
                
            else:
                if Self:
                        inputSeq = self.wordEmbed(inputSeq)
                        outputs, _ = self.rnn(inputSeq, encStates)
                        #                             outputs = F.dropout(outputs, self.dropout, training=self.training)
                        energy = self.projection(outputs)
                        weights = F.softmax(energy.squeeze(-1), dim=1)  
                        outputs = (outputs * weights.unsqueeze(-1)).sum(dim=1)
                        outputSize = outputs.size()                           
                        flatOutputs = outputs.view(-1, outputSize[2])
                        flatScores = self.outNet(flatOutputs)
                        flatLogProbs = self.logSoftmax(flatScores)
                        logProbs = flatLogProbs.view(outputSize[0], outputSize[1], -1)

                else:
                        inputSeq = self.wordEmbed(inputSeq)
                        outputs, _ = self.rnn(inputSeq, encStates)
                        outputs = F.dropout(outputs, self.dropout, training=self.training)
                        outputSize = outputs.size()
                        flatOutputs = outputs.view(-1, outputSize[2])
                        flatScores = self.outNet(flatOutputs)
                        flatLogProbs = self.logSoftmax(flatScores)
                        logProbs = flatLogProbs.view(outputSize[0], outputSize[1], -1)
                            
        return logProbs

    def forwardDecode(self,
                      encStates,
                      maxSeqLen=20,
                      inference='sample',
                      beamSize=1, topk=1, retLogProbs=False, gamma=0, delta=0,enc_outputs = None, att = False):
        '''
        Decode a sequence of tokens given an encoder state, using either
        sampling or greedy inference.

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            maxSeqLen : Maximum length of token sequence to generate
            inference : Inference method for decoding
                'sample' - Sample each word from its softmax distribution
                'greedy' - Always choose the word with highest probability
                           if beam size is 1, otherwise use beam search.
            beamSize  : Beam search width
            topk      : Return topk answers from beamsearch
            retLogProbs   : Return log probability of answer, only support
                            greedy inference

        Notes:
            * This function is not called during SL pre-training
            * Greedy inference is used for evaluation
            * Sampling is used in RL fine-tuning
        '''
        if inference == 'greedy' and beamSize > 1:
            # Use beam search inference when beam size is > 1
            if gamma != 0 and delta != 0:
                return self.diverseBeamSearchDecoder(encStates, beamSize, maxSeqLen, topk=topk, retLogProbs=retLogProbs,
                                                     gamma=gamma, delta=delta,enc_outputs = enc_outputs,att = att)
            else:
                return self.beamSearchDecoder(encStates, beamSize, maxSeqLen, topk=topk, 
                                              retLogProbs=retLogProbs,enc_outputs = enc_outputs,att = att)


        # Determine if cuda tensors are being used
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        self.samples = []
        maxLen = maxSeqLen + 1  # Extra <END> token
        batchSize = encStates[0].size(1)
        # Placeholder for filling in tokens at evert time step
        seq = th.LongTensor(batchSize, maxLen + 1)
        seq.fill_(self.endToken)
        seq[:, 0] = self.startToken
        seq = Variable(seq, requires_grad=False)

        # Initial state linked from encStates
        hid = encStates

        sampleLens = th.LongTensor(batchSize).fill_(0)
        # Tensors needed for tracking sampleLens
        unitColumn = th.LongTensor(batchSize).fill_(1)
        mask = th.ByteTensor(seq.size()).fill_(0)

        self.saved_log_probs = []
        logProbList = []

        # Generating tokens sequentially
        for t in range(maxLen - 1):
            emb = self.wordEmbed(seq[:, t:t + 1])
            # emb has shape  (batch, 1, embedSize)
            output, hid = self.rnn(emb, hid)
            # output has shape (batch, 1, rnnHiddenSize)
            scores = self.outNet(output.squeeze(1))
            logProb = self.logSoftmax(scores)

            # Explicitly removing padding token (index 0) and <START> token
            # (index -2) from logProbs so that they are never sampled.
            # This is allows us to keep <START> and padding token in
            # the decoder vocab without any problems in RL sampling.
            if t > 0:
                logProb = torch.cat([logProb[:, 1:-2], logProb[:, -1:]], 1)
            elif t == 0:
                # Additionally, remove <END> token from the first sample
                # to prevent the sampling of an empty sequence.
                logProb = logProb[:, 1:-2]
            
            # This also shifts end token index back by 1
            END_TOKEN_IDX = self.endToken - 1

            probs = torch.exp(logProb)
            if inference == 'sample':
                categorical_dist = Categorical(probs)
                sample = categorical_dist.sample()
                # Saving log probs for a subsequent reinforce call
                self.saved_log_probs.append(categorical_dist.log_prob(sample))
                sample = sample.unsqueeze(-1)
            elif inference == 'greedy':
                _, sample = torch.max(probs, dim=1, keepdim=True)
            else:
                raise ValueError(
                    "Invalid inference type: '{}'".format(inference))

            # Compensating for removed padding token prediction earlier
            sample = sample + 1  # Incrementing all token indices by 1

            self.samples.append(sample)
            seq.data[:, t + 1] = sample.data
            # Marking spots where <END> token is generated
            mask[:, t] = sample.data.eq(END_TOKEN_IDX)

            # Compensating for shift in <END> token index
            sample.data.masked_fill_(mask[:, t].unsqueeze(1), self.endToken)

            if retLogProbs:
                ansLogProbs, _ = torch.max(logProb, dim=1, keepdim=True)
                logProbList.append(ansLogProbs)
        

        mask[:, maxLen - 1].fill_(1)

        # Computing lengths of generated sequences
        for t in range(maxLen):
            # Zero out the spots where end token is reached
            unitColumn.masked_fill_(mask[:, t], 0)
            # Update mask
            mask[:, t] = unitColumn
            # Add +1 length to all un-ended sequences
            sampleLens = sampleLens + unitColumn
        

        # Keep mask for later use in RL reward masking
        self.mask = Variable(mask, requires_grad=False)

        if retLogProbs:
            revMask = 1 - self.mask[:, :-2].float()
            revMask = revMask.byte()
            logProbList = torch.cat(logProbList, 1)
            logProbList.masked_fill_(revMask, 0)
            # logProbSeq = torch.sum(logProbList, dim=1)

        # Adding <START> to generated answer lengths for consistency
        sampleLens = sampleLens + 1
        sampleLens = Variable(sampleLens, requires_grad=False)

        startColumn = sample.data.new(sample.size()).fill_(self.startToken)
        startColumn = Variable(startColumn, requires_grad=False)

        # Note that we do not add startColumn to self.samples itself
        # as reinforce is called on self.samples (which needs to be
        # the output of a stochastic function)
        gen_samples = [startColumn] + self.samples

        samples = torch.cat(gen_samples, 1)
        if retLogProbs:
            return samples, sampleLens, logProbList
        return samples, sampleLens

    def evalOptions(self, encStates, options, optionLens, scoringFunction):
        '''
        Forward pass a set of candidate options to get log probabilities

        Arguments:
            encStates : (H, C) Tuple of hidden and cell encoder states
            options   : (batchSize, numOptions, maxSequenceLength) sized
                        tensor with <START> and <END> tokens

            scoringFunction : A function which computes negative log
                              likelihood of a sequence (answer) given log
                              probabilities under an RNN model. Currently
                              utils.maskedNll is the only such function used.

        Output:
            A (batchSize, numOptions) tensor containing the score
            of each option sentence given by the generator
        '''
        batchSize, numOptions, maxLen = options.size()
        optionsFlat = options.contiguous().view(-1, maxLen)

        # Reshaping H, C for each option
        encStates = [x.unsqueeze(2).repeat(1,1,numOptions,1).\
                        view(self.numLayers, -1, self.rnnHiddenSize)
                        for x in encStates]

        logProbs = self.forward(encStates, inputSeq=optionsFlat)
        scores = scoringFunction(logProbs, optionsFlat, returnScores=True)
        return scores.view(batchSize, numOptions)

    def reinforce(self, reward):
        '''
        Compute loss using REINFORCE on log probabilities of tokens
        sampled from decoder RNN, scaled by input 'reward'.

        Note that an earlier call to forwardDecode must have been
        made in order to have samples for which REINFORCE can be
        applied. These samples are stored in 'self.saved_log_probs'.
        '''
        loss = 0
        # samples = torch.stack(self.samples, 1)
        # sampleLens = self.sampleLens - 1
        if len(self.saved_log_probs) == 0:
            raise RuntimeError("Reinforce called without sampling in Decoder")

        for t, log_prob in enumerate(self.saved_log_probs):
            loss += -1 * log_prob * (reward * (self.mask[:, t].float()))
        return loss

    def diverseBeamSearchDecoder(self, initStates, beamSize, maxSeqLen, topk=0, retLogProbs=False, gamma=0, delta=0,enc_outputs = None,att = False):
        '''
        Beam search for sequence generation

        Arguments:
            initStates - Initial encoder states tuple
            beamSize - Beam Size
            maxSeqLen - Maximum length of sequence to decode
            topk - Return k best answers, 0 for all results
        '''

        assert topk >= 0 and topk <= beamSize

        # use beam search for evaluation only
        assert self.training == False
        
        if self.wordEmbed.weight.is_cuda:
            th = torch.cuda
        else:
            th = torch

        LENGTH_NORM = True
        maxLen = maxSeqLen + 1  # Extra <END> token
        batchSize = initStates[0].size(1)

        startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
        backVector = th.LongTensor(beamSize)
        torch.arange(0, beamSize, out=backVector)
        backVector = backVector.unsqueeze(0).repeat(batchSize, 1)  # (batchSize, beamSize)
        
        tokenArange = th.LongTensor(self.vocabSize)
        torch.arange(0, self.vocabSize, out=tokenArange)
        tokenArange = Variable(tokenArange) # (vocabSize, )

        startTokenArray = Variable(startTokenArray)
        backVector = Variable(backVector)
        hiddenStates = initStates
        
        # Inits
        beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(
            self.endToken)
        
        beamTokensTable = Variable(beamTokensTable)
        backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
        backIndices = Variable(backIndices)
        
        # (batchSize, beamSize, 1)
        aliveVector = beamTokensTable[:, :, 0].eq(self.endToken).unsqueeze(2)

                
        for t in range(maxLen - 1):  # Beam expansion till maxLen
            if att:
                if t == 0:
                     # First column of beamTokensTable is generated from <START> token
                    emb = self.wordEmbed(startTokenArray)
            
                     # emb has shape (batchSize, 1, embedSize)
                    outputs,hiddenStates = self.rnn(emb, hiddenStates)
                     # output has shape (batchSize, 1, rnnHiddenSize)

                    outputs = F.dropout(outputs, self.dropout, training=self.training)
                    
                    e_outputs = self.attn(enc_outputs).unsqueeze(1).transpose(1,2)
                    
                    attn_energies = outputs.bmm(e_outputs)
                    attn_weights = self.my_log_softmax(attn_energies)
                    
                    context = attn_weights.bmm(e_outputs.transpose(1,2))
                    
                    output_context = torch.cat((outputs, context), 2)
                    output_context = self.concat(output_context)
                    output = F.tanh(output_context)
                    scores = self.outNet(output.squeeze(1))
                    logProbs = self.logSoftmax(scores)
                    # scores & logProbs has shape (batchSize, vocabSize)

                    # Find top beamSize logProbs
                    topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                    # beamTokensTable[:, :, 0] = topIdx.transpose(0, 1).data
                    beamTokensTable[:, :, 0] = topIdx.data
                    logProbSums = topLogProbs

                    # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                    hiddenStates = [
                        x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                        for x in hiddenStates
                    ]
                    hiddenStates = [
                        x.view(self.numLayers, -1, self.rnnHiddenSize)
                        for x in hiddenStates
                    ]
                    # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
                else:
                    # Subsequent columns are generated from previous tokens
                    emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                    # print(beamTokensTable.shape)
                    # emb has shape (batchSize, beamSize, embedSize)
                    # print(outputs.shape,emb.shape)
                    output, hiddenStates = self.rnn(emb.view(20, 1, self.embedSize), hiddenStates)
                    # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                    # print(outputs.shape)
                    outputs = F.dropout(outputs, self.dropout, training=self.training)
                    e_outputs = self.attn(enc_outputs).unsqueeze(1).transpose(1,2)
                    attn_energies = outputs.bmm(e_outputs.repeat(outputs.shape[0],1,1))
                    # print(outputs.shape,e_outputs.repeat(outputs.shape[0],1,1).shape,attn_energies.shape)
                    attn_weights = self.my_log_softmax(attn_energies)
                    context = attn_weights.bmm(e_outputs.transpose(1,2))
                    output_context = torch.cat((outputs, context), 2)
                    output_context = self.concat(output_context)
                    output = F.tanh(output_context)
                    scores = self.outNet(output.squeeze(1))
                    logProbsCurrent = self.logSoftmax(scores)
                    logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,self.vocabSize)
                    logProbsCurrent, logProbsSortedIdx = torch.sort(logProbsCurrent, dim=2, descending=True)
                    logProbsCurrent -= gamma * Variable(torch.arange(self.vocabSize).cuda())
                    _, logProbsSortedIdx2 = torch.sort(logProbsSortedIdx, dim=2)
                    logProbsCurrent = logProbsCurrent.gather(2, logProbsSortedIdx2)

                    logProbsCurrent, logProbsSortedIdx = torch.sort(logProbsCurrent, dim=1, descending=True)
                    logProbsCurrent = (logProbsCurrent.transpose(1, 2) - delta * Variable(torch.arange(beamSize).cuda())).transpose(1, 2)
                    _, logProbsSortedIdx2 = torch.sort(logProbsSortedIdx, dim=1)
                    logProbsCurrent = logProbsCurrent.gather(1, logProbsSortedIdx2)

                    if LENGTH_NORM:
                        # Add (current log probs / (t+1))
                        logProbs = logProbsCurrent * (aliveVector.float() /
                                                      (t + 1))
                        # Add (previous log probs * (t/t+1) ) <- Mean update
                        coeff_ = aliveVector.eq(0).float() + (
                            aliveVector.float() * t / (t + 1))
                        logProbs += logProbSums.unsqueeze(2) * coeff_
                    else:
                        # Add currrent token logProbs for alive beams only
                        logProbs = logProbsCurrent * (aliveVector.float())
                        # Add previous logProbSums upto t-1
                        logProbs += logProbSums.unsqueeze(2)

                    # Masking out along |V| dimension those sequence logProbs
                    # which correspond to ended beams so as to only compare
                    # one copy when sorting logProbs
                    mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                    mask_[:, :,
                          0] = 0  # Zeroing all except first row for ended beams
                    minus_infinity_ = torch.min(logProbs).data[0]
                    logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                    logProbs = logProbs.view(batchSize, -1) # (batchSize, beamSize * vocabSize)
                    tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).\
                                    repeat(batchSize,beamSize,1)
                    tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                    tokensArray = tokensArray.view(batchSize, -1) # (batchSize, beamSize * vocabSize)
                    backIndexArray = backVector.unsqueeze(2).\
                                    repeat(1,1,self.vocabSize).view(batchSize,-1) # (batchSize, beamSize * vocabSize)

                    topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                    logProbSums = topLogProbs
                    beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                    backIndices[:, :, t] = backIndexArray.gather(1, topIdx)


        # for t in range(maxLen - 1):  # Beam expansion till maxLen]
            else:
                if t == 0:
                    # First column of beamTokensTable is generated from <START> token
                    emb = self.wordEmbed(startTokenArray)
                    # emb has shape (batchSize, 1, embedSize)
                    output, hiddenStates = self.rnn(emb, hiddenStates)
                    # output has shape (batchSize, 1, rnnHiddenSize)
                    scores = self.outNet(output.squeeze(1))
                    logProbs = self.logSoftmax(scores)
                    # scores & logProbs has shape (batchSize, vocabSize)

                    # Find top beamSize logProbs
                    topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                    # beamTokensTable[:, :, 0] = topIdx.transpose(0, 1).data
                    beamTokensTable[:, :, 0] = topIdx.data
                    logProbSums = topLogProbs

                    # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                    hiddenStates = [
                        x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                        for x in hiddenStates
                    ]
                    hiddenStates = [
                        x.view(self.numLayers, -1, self.rnnHiddenSize)
                        for x in hiddenStates
                    ]
                    # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
                else:
                    # Subsequent columns are generated from previous tokens
                    emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                    # emb has shape (batchSize, beamSize, embedSize)
                    output, hiddenStates = self.rnn(
                        emb.view(-1, 1, self.embedSize), hiddenStates)
                    # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                    scores = self.outNet(output.squeeze())
                    logProbsCurrent = self.logSoftmax(scores)
                    
                    logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                                                           self.vocabSize)
                    logProbsCurrent, logProbsSortedIdx = torch.sort(logProbsCurrent, dim=2, descending=True)
                    logProbsCurrent -= gamma * Variable(torch.arange(self.vocabSize).cuda())
                    _, logProbsSortedIdx2 = torch.sort(logProbsSortedIdx, dim=2)
                    logProbsCurrent = logProbsCurrent.gather(2, logProbsSortedIdx2)

                    logProbsCurrent, logProbsSortedIdx = torch.sort(logProbsCurrent, dim=1, descending=True)
                    logProbsCurrent = (logProbsCurrent.transpose(1, 2) - delta * Variable(torch.arange(beamSize).cuda())).transpose(1, 2)
                    _, logProbsSortedIdx2 = torch.sort(logProbsSortedIdx, dim=1)
                    logProbsCurrent = logProbsCurrent.gather(1, logProbsSortedIdx2)

                    if LENGTH_NORM:
                        # Add (current log probs / (t+1))
                        logProbs = logProbsCurrent * (aliveVector.float() /
                                                      (t + 1))
                        # Add (previous log probs * (t/t+1) ) <- Mean update
                        coeff_ = aliveVector.eq(0).float() + (
                            aliveVector.float() * t / (t + 1))
                        logProbs += logProbSums.unsqueeze(2) * coeff_
                    else:
                        # Add currrent token logProbs for alive beams only
                        logProbs = logProbsCurrent * (aliveVector.float())
                        # Add previous logProbSums upto t-1
                        logProbs += logProbSums.unsqueeze(2)

                   
                    mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                    mask_[:, :,
                          0] = 0  # Zeroing all except first row for ended beams
                    minus_infinity_ = torch.min(logProbs).data[0]
                    logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                    logProbs = logProbs.view(batchSize, -1) # (batchSize, beamSize * vocabSize)
                    tokensArray = tokenArange.unsqueeze(0).unsqueeze(0).\
                                    repeat(batchSize,beamSize,1)
                    tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                    tokensArray = tokensArray.view(batchSize, -1) # (batchSize, beamSize * vocabSize)
                    backIndexArray = backVector.unsqueeze(2).\
                                    repeat(1,1,self.vocabSize).view(batchSize,-1) # (batchSize, beamSize * vocabSize)

                    topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                    logProbSums = topLogProbs
                    beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                    backIndices[:, :, t] = backIndexArray.gather(1, topIdx)

            # Detecting endToken to end beams
            aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
            # print(aliveVector)
            aliveBeams = aliveVector.data.long().sum()
            finalLen = t
            if aliveBeams == 0:
                break

        # Backtracking to get final beams
        beamTokensTable = beamTokensTable.data
        backIndices = backIndices.data

        # Keep this on when returning the top beam
        # RECOVER_TOP_BEAM_ONLY = True

        tokenIdx = finalLen
        backID = backIndices[:, :, tokenIdx]
        tokens = []
        while (tokenIdx >= 0):
            tokens.append(beamTokensTable[:,:,tokenIdx].\
                        gather(1, backID).unsqueeze(2))
            backID = backIndices[:, :, tokenIdx].\
                        gather(1, backID)
            tokenIdx = tokenIdx - 1

        tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
        tokens.reverse()
        tokens = torch.cat(tokens, 2)
        seqLens = tokens.ne(self.endToken).long().sum(dim=2)

        '''
        if RECOVER_TOP_BEAM_ONLY:
            # 'tokens' has shape (batchSize, beamSize, maxLen)
            # 'seqLens' has shape (batchSize, beamSize)
            tokens = tokens[:, 0]  # Keep only top beam
            seqLens = seqLens[:, 0]
        '''
        
        if topk > 1:
            tokens = tokens[:, :topk]
            seqLens = seqLens[:, :topk]
            logProbSums = logProbSums[:, :topk]
        else:
            tokens = tokens[:, 0]
            seqLens = seqLens[:, 0]
            logProbSums = logProbSums[:, 0]


        if retLogProbs:
            return Variable(tokens), Variable(seqLens), logProbSums
        else:
            return Variable(tokens), Variable(seqLens)

    def beamSearchDecoder(self, initStates, beamSize, maxSeqLen, topk=0, retLogProbs=False,enc_outputs = None,att = False):
            '''
            Beam search for sequence generation
            Arguments:
                initStates - Initial encoder states tuple
                beamSize - Beam Size
                maxSeqLen - Maximum length of sequence to decode
                topk - Return k best answers, 0 for all results
            '''

            assert topk >= 0 and topk <= beamSize

            # For now, use beam search for evaluation only
            assert self.training == False

            # Determine if cuda tensors are being used
            if self.wordEmbed.weight.is_cuda:
                th = torch.cuda
            else:
                th = torch

            LENGTH_NORM = True
            maxLen = maxSeqLen + 1  # Extra <END> token
            batchSize = initStates[0].size(1)

            startTokenArray = th.LongTensor(batchSize, 1).fill_(self.startToken)
            backVector = th.LongTensor(beamSize)
            torch.arange(0, beamSize, out=backVector)
            backVector = backVector.unsqueeze(0).repeat(batchSize, 1)  # (batchSize, beamSize)

            tokenArange = th.LongTensor(self.vocabSize)
            torch.arange(0, self.vocabSize, out=tokenArange)
            tokenArange = Variable(tokenArange)  # (vocabSize, )

            startTokenArray = Variable(startTokenArray)
            backVector = Variable(backVector)
            hiddenStates = initStates

            # Inits
            beamTokensTable = th.LongTensor(batchSize, beamSize, maxLen).fill_(
                self.endToken)
            beamTokensTable = Variable(beamTokensTable)
            backIndices = th.LongTensor(batchSize, beamSize, maxLen).fill_(-1)
            backIndices = Variable(backIndices)
            # if retLogProbs:
            #     backLogProbs = th.FloatTensor(batchSize, beamSize, maxLen).fill_(0)
            #     backLogProbs = Variable(backLogProbs)

            # (batchSize, beamSize, 1)
            aliveVector = beamTokensTable[:, :, 0].eq(self.endToken).unsqueeze(2)

            for t in range(maxLen - 1):  # Beam expansion till maxLen]
                if t == 0:
                    # First column of beamTokensTable is generated from <START> token
                    emb = self.wordEmbed(startTokenArray)
                    # emb has shape (batchSize, 1, embedSize)
                    output, hiddenStates = self.rnn(emb, hiddenStates)
                    # output has shape (batchSize, 1, rnnHiddenSize)
                    scores = self.outNet(output.squeeze(1))
                    logProbs = self.logSoftmax(scores)
                    # scores & logProbs has shape (batchSize, vocabSize)

                    # Find top beamSize logProbs
                    topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)
                    # beamTokensTable[:, :, 0] = topIdx.transpose(0, 1).data
                    beamTokensTable[:, :, 0] = topIdx.data
                    logProbSums = topLogProbs

                    # Repeating hiddenStates 'beamSize' times for subsequent self.rnn calls
                    hiddenStates = [
                        x.unsqueeze(2).repeat(1, 1, beamSize, 1)
                        for x in hiddenStates
                    ]
                    hiddenStates = [
                        x.view(self.numLayers, -1, self.rnnHiddenSize)
                        for x in hiddenStates
                    ]
                    # H_0 and C_0 have shape (numLayers, batchSize*beamSize, rnnHiddenSize)
                else:
                    # Subsequent columns are generated from previous tokens
                    emb = self.wordEmbed(beamTokensTable[:, :, t - 1])
                    # emb has shape (batchSize, beamSize, embedSize)
                    output, hiddenStates = self.rnn(
                        emb.view(-1, 1, self.embedSize), hiddenStates)
                    # output has shape (batchSize*beamSize, 1, rnnHiddenSize)
                    scores = self.outNet(output.squeeze())
                    logProbsCurrent = self.logSoftmax(scores)
                    # logProbs has shape (batchSize*beamSize, vocabSize)
                    # NOTE: Padding token has been removed from generator output during
                    # sampling (RL fine-tuning). However, the padding token is still
                    # present in the generator vocab and needs to be handled in this
                    # beam search function. This will be supported in a future release.
                    logProbsCurrent = logProbsCurrent.view(batchSize, beamSize,
                                                           self.vocabSize)

                    if LENGTH_NORM:
                        # Add (current log probs / (t+1))
                        logProbs = logProbsCurrent * (aliveVector.float() /
                                                      (t + 1))
                        # Add (previous log probs * (t/t+1) ) <- Mean update
                        coeff_ = aliveVector.eq(0).float() + (
                                aliveVector.float() * t / (t + 1))
                        logProbs += logProbSums.unsqueeze(2) * coeff_
                    else:
                        # Add currrent token logProbs for alive beams only
                        logProbs = logProbsCurrent * (aliveVector.float())
                        # Add previous logProbSums upto t-1
                        logProbs += logProbSums.unsqueeze(2)

                    # Masking out along |V| dimension those sequence logProbs
                    # which correspond to ended beams so as to only compare
                    # one copy when sorting logProbs
                    mask_ = aliveVector.eq(0).repeat(1, 1, self.vocabSize)
                    mask_[:, :,
                    0] = 0  # Zeroing all except first row for ended beams
                    minus_infinity_ = torch.min(logProbs).data[0]
                    logProbs.data.masked_fill_(mask_.data, minus_infinity_)

                    logProbs = logProbs.view(batchSize, -1)  # (batchSize, beamSize * vocabSize)
                    tokensArray = tokenArange.unsqueeze(0).unsqueeze(0). \
                        repeat(batchSize, beamSize, 1)
                    tokensArray.masked_fill_(aliveVector.eq(0), self.endToken)
                    tokensArray = tokensArray.view(batchSize, -1)  # (batchSize, beamSize * vocabSize)
                    backIndexArray = backVector.unsqueeze(2). \
                        repeat(1, 1, self.vocabSize).view(batchSize, -1)  # (batchSize, beamSize * vocabSize)

                    topLogProbs, topIdx = logProbs.topk(beamSize, dim=1)

                    logProbSums = topLogProbs
                    beamTokensTable[:, :, t] = tokensArray.gather(1, topIdx)
                    backIndices[:, :, t] = backIndexArray.gather(1, topIdx)

                # Detecting endToken to end beams
                aliveVector = beamTokensTable[:, :, t:t + 1].ne(self.endToken)
                # print(aliveVector)
                aliveBeams = aliveVector.data.long().sum()
                finalLen = t
                if aliveBeams == 0:
                    break

            # Backtracking to get final beams
            beamTokensTable = beamTokensTable.data
            backIndices = backIndices.data

            # Keep this on when returning the top beam
            # RECOVER_TOP_BEAM_ONLY = True

            tokenIdx = finalLen
            backID = backIndices[:, :, tokenIdx]
            tokens = []
            while (tokenIdx >= 0):
                tokens.append(beamTokensTable[:, :, tokenIdx]. \
                              gather(1, backID).unsqueeze(2))
                backID = backIndices[:, :, tokenIdx]. \
                    gather(1, backID)
                tokenIdx = tokenIdx - 1

            tokens.append(startTokenArray.unsqueeze(2).repeat(1, beamSize, 1).data)
            tokens.reverse()
            tokens = torch.cat(tokens, 2)
            seqLens = tokens.ne(self.endToken).long().sum(dim=2)

            '''
            if RECOVER_TOP_BEAM_ONLY:
                # 'tokens' has shape (batchSize, beamSize, maxLen)
                # 'seqLens' has shape (batchSize, beamSize)
                tokens = tokens[:, 0]  # Keep only top beam
                seqLens = seqLens[:, 0]
            '''

            if topk > 1:
                tokens = tokens[:, :topk]
                seqLens = seqLens[:, :topk]
                logProbSums = logProbSums[:, :topk]
            else:
                tokens = tokens[:, 0]
                seqLens = seqLens[:, 0]
                logProbSums = logProbSums[:, 0]

            if retLogProbs:
                return Variable(tokens), Variable(seqLens), logProbSums
            else:
                return Variable(tokens), Variable(seqLens)

