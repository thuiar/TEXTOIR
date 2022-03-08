import scipy.spatial.distance as spd
import scipy as sp
import numpy as np
try:
    from .libMR import libmr
except ImportError:
    print ("LibMR not installed or libmr.so not found")
    print ("Install libmr: cd libMR/; ./compile.sh")
    
def computeOpenMaxProbability(openmax_fc8, openmax_score_u, n_classes):
    """ Convert the scores in probability value using openmax
    
    Input:
    ---------------
    openmax_fc8 : modified FC8 layer from Weibull based computation
    openmax_score_u : degree
    Output:
    ---------------
    modified_scores : probability values modified using OpenMax framework,
    by incorporating degree of uncertainity/openness for a given class
    
    """
    prob_scores, prob_unknowns = [], []
    channel_scores, channel_unknowns = [], []
    for category in range(n_classes):
        channel_scores += [sp.exp(openmax_fc8[category])]

    total_denominator = sp.sum(sp.exp(openmax_fc8[:])) + sp.exp(sp.sum(openmax_score_u[:]))

    prob_scores += [channel_scores / total_denominator]

    prob_unknowns += [sp.exp(sp.sum(openmax_score_u[:]))/total_denominator]
       
    prob_scores = sp.asarray(prob_scores)
    prob_unknowns = sp.asarray(prob_unknowns) 
    
    scores = sp.mean(prob_scores, axis = 0)
    unknowns = sp.mean(prob_unknowns, axis=0)
    modified_scores =  scores.tolist() + [unknowns]
    
    return modified_scores

def compute_distance(MAV, query_channel, distance_type):
    
    if distance_type == 'eucos':
        query_distance = spd.euclidean(MAV, query_channel) / 200.  + spd.cosine(MAV, query_channel) / 200.
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(MAV, query_channel)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(MAV, query_channel)
    else:
        print ("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance

def weibull_tailfitting(mean_vecs, distance_values, num_labels, 
                        tailsize = 20):
    
    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    for category in range(num_labels):
        weibull_model[category] = {}
        distances = distance_values[category]
        means = mean_vecs[category]
        weibull_model[category]['distances'] = distances
        weibull_model[category]['mean_vec'] = means
        
        mr = libmr.MR()
        
        tailtofit = distances[-tailsize:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] = mr

    return weibull_model

def query_weibull(category_name, weibull_model):
    """ Query through dictionary for Weibull model.
    Return in the order: [mean_vec, distances, weibull_model]
    
    Input:
    ------------------------------
    category_name : name of ImageNet category in WNET format. E.g. n01440764
    weibull_model: dictonary of weibull models for 
    """

    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec']]
    category_weibull += [weibull_model[category_name]['distances']]
    category_weibull += [weibull_model[category_name]['weibull_model']]

    return category_weibull 

def recalibrate_scores(weibull_model, num_labels, textarr, layer = 'fc8', alpharank = 5, distance_type = 'eucos'):
    
    txtlayer = textarr[layer]
    ranked_list = textarr['scores'].argsort().ravel()[::-1]
    alpha_weights = [((alpharank+1) - i)/float(alpharank) for i in range(1, alpharank+1)]
    
    ranked_alpha = sp.zeros(num_labels)

    for i in range(len(alpha_weights)): 
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    
    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    openmax_fc8, openmax_score_u = [], []

    cha_scores = txtlayer
    openmax_fc8_channel = []
    openmax_fc8_unknown = []
                            
    for categoryid in range(num_labels):
        # get distance between current channel and mean vector
        category_weibull = query_weibull(categoryid, weibull_model)
        distance = compute_distance(txtlayer, category_weibull[0], distance_type)                  
        
        # obtain w_score for the distance and compute probability of the distance
        # being unknown wrt to mean training vector and channel distances for
        # category and channel under consideration
        wscore = category_weibull[2].w_score(distance)
        
        modified_fc8_score = cha_scores[categoryid] * (1 - wscore * ranked_alpha[categoryid])
        openmax_fc8_channel += [modified_fc8_score]
        openmax_fc8_unknown += [cha_scores[categoryid] - modified_fc8_score]
                          
    openmax_fc8 = openmax_fc8_channel
    openmax_score_u = openmax_fc8_unknown
    openmax_fc8 = sp.asarray(openmax_fc8)
    openmax_score_u = sp.asarray(openmax_score_u)
    
    # Pass the recalibrated fc8 scores for the image into openmax    
    openmax_prob = computeOpenMaxProbability(openmax_fc8, openmax_score_u, num_labels)
    softmax_prob = textarr['scores'].ravel() 
    
    return sp.asarray(openmax_prob), sp.asarray(softmax_prob)
