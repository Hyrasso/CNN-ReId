En utilisant tensorflow.keras implémenter l’algorithme de ré-identification décrit dans l’article Lin et al. - Improving Person Re-identification by Attribute and Identity Learning - 2019
- En utilisant un resnet50 pré-entrainé sur imagenet, implémenter l’algorithme de classification des attributs
- Implémenter un premier algorithme de ré-identification en combinant la classification d’attributs avec une couche fully-connected supplémentaire pour l’identification
- Augmenter les données du dataset lors de l’apprentissage selon le protocole décrit en page 7 de l’article (“Randomly cropping and horizontal flipping are applied on the input images during training.”)
- Implémenter un système permettant de comparer 2 vecteurs de features. (La distance euclidienne est un bon point de départ.
- Bonus: Implémenter le module de re-pondération des attributs “Attributes Re-Weighting”
