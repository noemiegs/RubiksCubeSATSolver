# Résolution du cube 2x2

## Remarques
On peut remarquer que tourner la face gauche du cube est equivalent à tourner celle de droite dans l'autre sens.
De même pour toutes les faces.

On peut alors restreindre l'ensemble des actions à trois faces (disons : Gauche, Bas et Derrière) avec pour chacune :
<ol>
    <li>Sens horaire</li>
    <li>Demi-tour</li>
    <li>Sens anti-horaire</li>
</ol>

On peut alors remarquer que le cube de devant, en haut à droite est constament fixe (ce qui élimine une variable pour notre problème).

## Notations

### Les étapes
On note $t_{max}$ le nombre d'étapes pour résoudre le Rubik's Cube

On note $T = \{0, ..., t_{max}\}$ et $T^* = \{1, ..., t_{max}\}$

### Les actions
On note $a_{f, d}(t)$ l'action de tourner la face $f \in F = \{Left, Bottom, Back\}$ dans la direction $d \in D = \{Clockwise, Halfturn, Counterclockwise\}$ à l'étape $t \in T^*$

### Le cube
Le Rubik's Cube est representé par un ensemble de cube ayant un $id \in C = \{1, ..., 7\}$ et une orientation $o \in O = \{0, 1, 2\}$ (l'orientation $o = 0$ étant celle lorsque le Rubik's Cube est fini)

On note $x_{c, id}(t)$ la variable booléenne indiquant si le cube ayant l'identifiant $id \in C$ est sur la position $c \in C$ à l'étape $t \in T$ (le cube étant à la bonne place ssi $x_{id, id}(t)$)

Et $\theta_{c, o}(t)$ la variable booléenne indiquant si le cube à la position $c \in C$ est dans l'orientation $o \in O$ à l'étape $t \in T$

## Conditions d'arrêt

### Positions

L'$id$ étant définie en fonction de la position par défaut on doit avoir :
$$
\forall \ c, id \in C :
    \begin{cases}
        x_{c, id}(t_{max}) \ \ \text{si} \ \ c = id \\
        \neg x_{c, id}(t_{max}) \ \ \text{sinon} \\
    \end{cases}
$$

### Orientations

Les orientations par défaut étant $o = 0$ on doit avoir :
$$
\forall \ c \in C :
    \begin{cases}
        \theta_{c, 0}(t_{max}) \\
        \neg \theta_{c, 1}(t_{max}) \\
        \neg \theta_{c, 2}(t_{max})
    \end{cases}
$$

## Transitions

Les transitions changent à la fois l'orientation et la postion des cubes qu'elles affecte.

On notera $c'$ la position et $o'$ l'orientation du cube $c$ après la rotation.

On a donc $\forall \ t \in T^*$ :

Pour les positions :
- $\forall \ f \in F, \ \forall \ d \in D, \ \forall \ c, id \in C$ :
$$a_{f, d}(t) \Rightarrow \Big( x_{c', id}(t) = x_{c, id}(t - 1) \Big)$$

Soit :
$$
\begin{cases}
    x_{c', id}(t) \ \vee \ \neg x_{c, id}(t - 1) \ \vee \ \neg a_{f, d}(t) \\
    \neg x_{c', id}(t) \ \vee \ x_{c, id}(t - 1) \ \vee \ \neg a_{f, d}(t) \\
\end{cases}
$$

Pour les orientations :
- $\forall \ f \in F, \ \forall \ c \in C, \ \forall \ o \in O$ :
$$
\left( \bigvee_{d \in D} a_{f, d}(t) \right) \Rightarrow \Big( \theta_{c', o'}(t) = \theta_{c, o}(t - 1) \Big) \\
$$

Soit :
$$
\bigwedge_{d \in D}
    \begin{cases}
        \theta_{c', o'}(t) \ \vee \ \neg \theta_{c, o}(t - 1) \ \vee \ \neg a_{f, d}(t) \\
        \neg \theta_{c', o'}(t) \ \vee \ \theta_{c, o}(t - 1) \ \vee \ \neg a_{f, d}(t) \\
    \end{cases}
$$

## Contraintes

On ne peut évidemment pas faire plusieurs rotation à la fois d'où :
$$
\forall \ t \in T^* : \bigwedge_{
    \substack{
        (f, d), (f', d') \in F \times D\\
        f, d \ < \ f', d'\\
    }
} \neg a_{f, d}(t) \vee \neg a_{f', d'}(t)$$

Pour simplifier les transitions on force également le solveur à prendre une action à chaque étape :
$$
\forall \ t \in T^* : \bigvee_{(f, d) \in F \times D} a_{f, d}(t)
$$
