import numpy as np
import pandas as pd

def vision_kmeans(centroids, X, labels, save_path):
    epoch = 0
    
    num_classes = len(np.unique(labels))
    print('num_classes', num_classes)
    X_out = pd.DataFrame()
    X_out_center = pd.DataFrame()
    X_outwithcenter = pd.DataFrame()
    X_out = pd.DataFrame(X, index = labels)
    
    X_out_center = pd.DataFrame(centroids) #cluster centers
    print('00000000000000', X_out_center.index)
    
    #将中心放入到数据中，一并tsne，不能单独tsne
    X_outwithcenter=X_out.append(X_out_center)
    print('111111111111111', X_outwithcenter.index)
    #用TSNE进行数据降维并展示聚类结果
    from sklearn.manifold import TSNE
    tsne = TSNE()
    tsne.fit_transform(X_outwithcenter) #进行数据降维,并返回结果
    print('2222222222222', X_outwithcenter.index)
    X_tsne = pd.DataFrame(tsne.embedding_, index = X_outwithcenter.index)
    #将index化成原本的数据的index，tsne后index会变化

    import matplotlib.pyplot as plt
        
    color_list = [c[1] for c in cnames.items()]

    for i in range(num_classes):
        d = X_tsne[X_tsne.index == i]     #找出聚类类别为0的数据对应的降维结果
        if i != num_classes - 1:
            if i == 1:
                plt.scatter(d[0], d[1],c='black',
                            marker='.',s=5)
        # else:
        #   plt.scatter(d[0], d[1],c='gray',
        #                 marker='.',s=5)  
        
                     
    #取中心点，画出
    d = X_tsne.tail(num_classes)
    plt.scatter(d[0], d[1], c='red',s=20,
                    marker='*'              )

    plt.savefig('class_2.png',dpi=300)
    #plt.show()
    plt.clf()

cnames = {
            'aliceblue': '#F0F8FF',

            'antiquewhite': '#FAEBD7',

            'aqua': '#00FFFF',

            'aquamarine': '#7FFFD4',

            'azure': '#F0FFFF',

            'beige': '#F5F5DC',

            'bisque': '#FFE4C4',

            'black': '#000000',

            'blanchedalmond': '#FFEBCD',

            'blue': '#0000FF',

            'blueviolet': '#8A2BE2',

            'brown': '#A52A2A',

            'burlywood': '#DEB887',

            'cadetblue': '#5F9EA0',

            'chartreuse': '#7FFF00',

            'chocolate': '#D2691E',

            'coral': '#FF7F50',

            'cornflowerblue': '#6495ED',

            'cornsilk': '#FFF8DC',

            'crimson': '#DC143C',

            'cyan': '#00FFFF',

            'darkblue': '#00008B',

            'darkcyan': '#008B8B',

            'darkgoldenrod': '#B8860B',

            'darkgray': '#A9A9A9',

            'darkgreen': '#006400',

            'darkkhaki': '#BDB76B',

            'darkmagenta': '#8B008B',

            'darkolivegreen': '#556B2F',

            'darkorange': '#FF8C00',

            'darkorchid': '#9932CC',

            'darkred': '#8B0000',

            'darksalmon': '#E9967A',

            'darkseagreen': '#8FBC8F',

            'darkslateblue': '#483D8B',

            'darkslategray': '#2F4F4F',

            'darkturquoise': '#00CED1',

            'darkviolet': '#9400D3',

            'deeppink': '#FF1493',

            'deepskyblue': '#00BFFF',

            'dimgray': '#696969',

            'dodgerblue': '#1E90FF',

            'firebrick': '#B22222',

            'floralwhite': '#FFFAF0',

            'forestgreen': '#228B22',

            'fuchsia': '#FF00FF',

            'gainsboro': '#DCDCDC',

            'ghostwhite': '#F8F8FF',

            'gold': '#FFD700',

            'goldenrod': '#DAA520',

            'green': '#008000',

            'greenyellow': '#ADFF2F',

            'honeydew': '#F0FFF0',

            'hotpink': '#FF69B4',

            'indianred': '#CD5C5C',

            'indigo': '#4B0082',

            'ivory': '#FFFFF0',

            'khaki': '#F0E68C',

            'lavender': '#E6E6FA',

            'lavenderblush': '#FFF0F5',

            'lawngreen': '#7CFC00',

            'lemonchiffon': '#FFFACD',

            'lightblue': '#ADD8E6',

            'lightcoral': '#F08080',

            'lightcyan': '#E0FFFF',

            'lightgoldenrodyellow': '#FAFAD2',

            'lightgreen': '#90EE90',

            'lightgray': '#D3D3D3',

            'lightpink': '#FFB6C1',

            'lightsalmon': '#FFA07A',

            'lightseagreen': '#20B2AA',

            'lightskyblue': '#87CEFA',

            'lightslategray': '#778899',

            'lightsteelblue': '#B0C4DE',

            'lightyellow': '#FFFFE0',

            'lime': '#00FF00',

            'limegreen': '#32CD32',

            'linen': '#FAF0E6',

            'magenta': '#FF00FF',

            'maroon': '#800000',

            'mediumaquamarine': '#66CDAA',

            'mediumblue': '#0000CD',

            'mediumorchid': '#BA55D3',

            'mediumpurple': '#9370DB',

            'mediumseagreen': '#3CB371',

            'mediumslateblue': '#7B68EE',

            'mediumspringgreen': '#00FA9A',

            'mediumturquoise': '#48D1CC',

            'mediumvioletred': '#C71585',

            'midnightblue': '#191970',

            'mintcream': '#F5FFFA',

            'mistyrose': '#FFE4E1',

            'moccasin': '#FFE4B5',

            'navajowhite': '#FFDEAD',

            'navy': '#000080',

            'oldlace': '#FDF5E6',

            'olive': '#808000',

            'olivedrab': '#6B8E23',

            'orange': '#FFA500',

            'orangered': '#FF4500',

            'orchid': '#DA70D6',

            'palegoldenrod': '#EEE8AA',

            'palegreen': '#98FB98',

            'paleturquoise': '#AFEEEE',

            'palevioletred': '#DB7093',

            'papayawhip': '#FFEFD5',

            'peachpuff': '#FFDAB9',

            'peru': '#CD853F',

            'pink': '#FFC0CB',

            'plum': '#DDA0DD',

            'powderblue': '#B0E0E6',

            'purple': '#800080',

            'red': '#FF0000',

            'rosybrown': '#BC8F8F',

            'royalblue': '#4169E1',

            'saddlebrown': '#8B4513',

            'salmon': '#FA8072',

            'sandybrown': '#FAA460',

            'seagreen': '#2E8B57',

            'seashell': '#FFF5EE',

            'sienna': '#A0522D',

            'silver': '#C0C0C0',

            'skyblue': '#87CEEB',

            'slateblue': '#6A5ACD',

            'slategray': '#708090',

            'snow': '#FFFAFA',

            'springgreen': '#00FF7F',

            'steelblue': '#4682B4',

            'tan': '#D2B48C',

            'teal': '#008080',

            'thistle': '#D8BFD8',

            'tomato': '#FF6347',

            'turquoise': '#40E0D0',

            'violet': '#EE82EE',

            'wheat': '#F5DEB3',

            'white': '#FFFFFF',

            'whitesmoke': '#F5F5F5',

            'yellow': '#FFFF00',

            'yellowgreen': '#9ACD32',

            'gray': '#808080',
            
}

if __name__ == '__main__':

    centroids = np.load('centroids.npy')
    feats = np.load('feats.npy')
    y_true = np.load('y_true.npy')
    vision_kmeans(centroids, feats, y_true)
    