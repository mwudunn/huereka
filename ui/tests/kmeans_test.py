from sklearn.cluster import KMeans

def compress_im(im, n_clusters, samples=1000):
    '''Compresses color image by finding the top n_groups of pixels, 
    then replacing all pixels with their closest neighbor in that group.
    
    Don't worry about the specifics of this function.
    '''
    # house keeping code to massage the data formats
    if im.dtype == np.uint8:
        im = im.astype(np.float64)/255        
    shape = im.shape    
    im_X = im.reshape(shape[0]*shape[1], 3)
    im_X_to_sample = im_X.copy()
    shuffle(im_X_to_sample)
    im_X_sampled = im_X_to_sample[:samples]
    
    # using KMeans algorithm to find n clusters
    kmeans = KMeans(n_clusters=n_clusters).fit(im_X_sampled)
    
    # reconstructing compressed image with the found clusters
    centers = kmeans.cluster_centers_
    group_assignments = kmeans.predict(im_X)
    im_Y = np.take(centers, group_assignments, axis=0)
    im_compressed = im_Y.reshape(shape[0], shape[1], 3)
    return im_compressed