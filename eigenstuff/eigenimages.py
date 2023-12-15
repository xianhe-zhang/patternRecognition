# Bruce Maxwell
# CS 5330 F23
# Tool for building a set of eigenimages from a collection of images
#
#
# 1: run the code and make sure it works, you may need to install openCV with pip
# 2: pick a set of images of your own that are coherent in some way
# 3: crop the images according to some standard of your own
# 4: put the images into a new directory
# 5: run the program with the new directory as the argument
# 6: modify the code so (after the existing code) it reads a new image (not in the directory) and does the following
#    A) display the original image
#    B) subtract the mean image from the original image to get the differential image
#    C) display the differential image
#    D) project the differential image into eigenspace by taking the dot product with the first N eigenvectors
#    E) print the projected coefficients
#    F) reproject from the coeffficients back to a differential image
#    G) display the reprojected differential image
#    H) add the mean image to the reprojected differential image to get the reprojected original
#    I) display the reprojected original image
#
# Try step six with an image that is similar to the ones you used to create the space
# Try step six with an image that is not similar to the ones you used to create the space
"""
Modified by Xianhe Zhang
To run this project, simply use `python3 eigenimages ./test_s.jpg ./test_ns.jpg`
"""
import sys
import os
import numpy as np
import cv2

# TODO: preprocessImage() Function: crop -> resize -> reshape -> format
def preprocessImage(image, rs, cs):
    image = image[:,:,1]
    image = cv2.resize(image, (cs, rs), interpolation=cv2.INTER_AREA)
    image = np.reshape(image, image.shape[0]*image.shape[1])
    image = image.astype("float32")
    print('1')
    return image


# reshapes and normalizes a 1-D column image, works for eigenvectors,
# differential images, and original images
def viewColumnImage( column, orgrows, orgcols, name ):

    src = np.reshape( column, (orgrows, orgcols) )
    minval = np.min(src)
    maxval = np.max(src)

    src = 255 * ( (src - minval) / (maxval - minval) )
    view = src.astype( "uint8" )
    cv2.imshow( name, view )
    cv2.imwrite(name+'.jpg', view)
    

def main(argv):

    orgrows = 0
    orgcols = 0

    # check for a directory path
    if len(argv) < 4:
        print("usage: python %s <directory name> <similar_image> <not_similar_image>" % (argv[0]))
        return

    # grab the directory path
    srcdir = argv[1]
    similarPath = argv[2]
    not_similarPath = argv[3]
    print(f'similarPath: {similarPath}')
    print(f'not_similarPath: {not_similarPath}')
    # open the directory and get a file listing
    filelist = os.listdir( srcdir )

    buildmtx = True
    print("Processing srcdir")

    for filename in filelist:

        print("Processing file %s" % (filename) )

        suffix = filename.split(".")[-1]

        if not ('pgm' in suffix or 'tif' in suffix or 'jpg' in suffix or 'png' in suffix ):
            continue

        src = cv2.imread( srcdir + "/" + filename )

        # make the image a single channel
        src = src[:, :, 1]

        # resize the long side of the image to 160
        resizeFactor =  160 / max( src.shape[0], src.shape[1] )

        if orgrows == 0:
            # resize, the resize function takes in (columns, rows) as a Size argument
            src = cv2.resize( src, (int(src.shape[1]*resizeFactor), int(src.shape[0]*resizeFactor)), interpolation=cv2.INTER_AREA )
        else:
            # ensure that all subsequent images end up the same size as the first one
            src = cv2.resize( src, (orgcols, orgrows), interpolation=cv2.INTER_AREA )

        # resize the image to a 1-D single vector
        newImg = np.reshape( src, src.shape[0]*src.shape[1] )
        newImg = newImg.astype("float32")

        # could probably normalize the image to sum to one
        # newImg /= np.sum(newImg)

        if buildmtx:
            # first image
            Amtx = newImg
            buildmtx = False
            orgrows = src.shape[0]
            orgcols = src.shape[1]

        else:
            Amtx = np.vstack( (Amtx, newImg) )

    # all of the images are in Amtx, which has 1 row and rows*cols columns
    # compute mean image
    meanvec = np.mean( Amtx, axis=0 )

    # build the differential data matrix and transpose so each image is a column
    Dmtx = Amtx - meanvec
    Dmtx = Dmtx.T

    # compute the singular value decomposition
    # N images
    #
    # U is rows*cols x N, columns of U are the eigenvectors of Dmtx
    # V is N x N, rows of V would be the eigenvectors of the rows of Dmtx
    # s contains the singular values which are related to the eigenvalues
    U, s, V = np.linalg.svd( Dmtx, full_matrices = False )

    eval = s**2 / (Dmtx.shape[0] - 1)

    print("top 10: ", eval[0:10] )

    for i in range(8):
        name = "eigenvector %d" % (i)
        viewColumnImage( U[:,i], orgrows, orgcols, name )
        cv2.moveWindow( name, i * 170, 200 )

    # look at the mean vector
    viewColumnImage( meanvec, orgrows, orgcols, "Mean image" )
    cv2.moveWindow( "Mean image", 100, 0)

    similar_img = cv2.imread(similarPath)
    similar_img = preprocessImage(similar_img, orgrows, orgcols)
    not_similar_img = cv2.imread(not_similarPath)
    not_similar_img = preprocessImage(not_similar_img, orgrows, orgcols)


    # project a set of images onto the first 6 eigenvectors
    # 0, 6 are digit 7; 1, 2 are digit 0
    position = 0
    for index, image in enumerate([similar_img, not_similar_img]):
        name = 'Similar' if index == 0 else 'Not Similar'
    
        # A - display the original image
        viewColumnImage(image, orgrows, orgcols, "Original " + name)
        cv2.moveWindow("Original " + name, position * 170, 400)
        
        # B - Subtract the mean image
        image = image - meanvec

        # C - show the differential image
        viewColumnImage(image, orgrows, orgcols, "Differential " + name)
        cv2.moveWindow("Differential " + name, (position + 1) * 170, 400)

        # D - take the dot product of the differential image and the first six eigenvectors
        projection = np.dot( image.T, U[:,0:15])

        # E - print the coefficients
        toprint = "Image %d: " % (index)
        for j in range(len(projection)):
            toprint += "%7.1f  " % (projection[j])
        print(toprint)

        # F - reproject from the six coefficients back to a new image
        recreated = projection[0] * U[:,0]
        for j in range(1,len(projection)):
            recreated += projection[j] * U[:,j]  # sum the coefficients times the corresopnding eigenvectors

        # G - show the recreated differential image, note less noise
        name = "Recreated %d" % (index)
        viewColumnImage( recreated, orgrows, orgcols, name )
        cv2.moveWindow(name, position * 170, 600 )

        # H - show the recreated original image (after adding back the mean image) note less noise
        name = "Recreated Original %d" % (index)
        viewColumnImage( recreated + meanvec, orgrows, orgcols, name )
        cv2.moveWindow(name, (position+1) * 170, 400 )

        # I - show the original image
        name = "Original %d" % (index)
        viewColumnImage( Amtx[index,:], orgrows, orgcols, name )
        cv2.moveWindow(name, (position+1) * 170, 600 )
        position += 2

    
    cv2.waitKey(0)
    

if __name__ == "__main__":
    main(sys.argv)



