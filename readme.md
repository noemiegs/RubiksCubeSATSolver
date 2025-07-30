# Résolution du cube 2x2

## Remarques
On peut remarquer que tourner la face droite du cube est equivalent à tourner celle de gauche dans l'autre sens.
De même pour toutes les faces.

On peut alors restreindre l'ensemble des actions à trois faces (disons : Droite, Bas et Derrière) avec pour chacune :
<ol>
    <li>Sens horaire</li>
    <li>Demi-tour</li>
    <li>Sens anti-horaire</li>
</ol>

On peut alors remarquer que le cube de devant, en haut à gauche est constament fixe (ce qui élimine une variable pour notre problème).

## Notations

### Les étapes
On note $t_{max}$ le nombre d'étapes pour résoudre le Rubik's Cube

On note $T = \{0, ..., t_{max}\}$ et $T^* = \{1, ..., t_{max}\}$

### Les actions
On note $a_{f, d}(t)$ l'action de tourner la face $f \in F = \{Right, Bottom, Back\}$ dans la direction $d \in D = \{Clockwise, Halfturn, Counterclockwise\}$ à l'étape $t \in T^*$

### Le cube
Le Rubik's Cube est representé par un ensemble de cube ayant un $id \in C = \{1, ..., 7\}$ et une orientation $o \in O = \{0, 1, 2\}$ (l'orientation $o = 0$ étant celle lorsque le Rubik's Cube est fini)

On note $x_{c, id}(t)$ la variable booléenne indiquant si le cube ayant l'identifiant $id \in C$ est sur la position $c \in C$ à l'étape $t \in T$ (le cube étant à la bonne place ssi $x_{id, id}(t)$)

Et $\theta_{c, o}(t)$ la variable booléenne indiquant si le cube à la position $c \in C$ est dans l'orientation $o \in O$ à l'étape $t \in T$

### Fonctions rotation

On note : $r_x : F \times D \times C \rightarrow C$, la fonction qui à chaque position associe la position après la rotation

On note : $r_\theta : F \times D \times C \times O \rightarrow O$, la fonction qui à chaque couple de position orientation associe l'orientation après la rotation

### Autres

Par abus de notation, on notera $c' = r_x(f, d, c)$ et $o' = r_\theta(f, d, c, o)$

Pour chaque face $f \in F$, on notera $C_f$ l'esemble des cube affecté par la rotation

On notera la permutation entre $i$ et $j$
$$s_{i, j} :
\begin{cases}
    O \rightarrow O \\
    o \mapsto
    \begin{cases}
        i & \text{si} \quad o = j \\
        j & \text{si} \quad o = i \\
        o & \text{sinon}
    \end{cases}
\end{cases}$$

On notera $c_x$, $c_y$ et $c_z$ les coordonnées du cube $c$

Enfin, on notera $g$ la fonction permettant d'obtenir les coordonnées du cube $c$ : $g(c) = c_x, c_y, c_z$. Et son inverse $g^{-1}(c_x, c_y, c_z) = c$

Soient $c \in C$ et $o \in O$.

### Positions

- Rotation de la face droite :

$$
r_x(\text{Right}, \text{Clockwise}, c) =
\begin{cases}
    g^{-1}(c_x, c_z, 1 - c_y) & \text{si } c \in C_{\text{Right}} \\
    c & \text{sinon}
\end{cases}
$$

- Rotation de la face du bas :

$$
r_x(\text{Bottom}, \text{Clockwise}, c) =
\begin{cases}
    g^{-1}(1 - c_z, c_y, c_x) & \text{si } c \in C_{\text{Bottom}} \\
    c & \text{sinon}
\end{cases}
$$

- Rotation de la face arrière :

$$
r_x(\text{Back}, \text{Clockwise}, c) =
\begin{cases}
    g^{-1}(c_y, 1 - c_x, c_z) & \text{si } c \in C_{\text{Back}} \\
    c & \text{sinon}
\end{cases}
$$

Les autres directions se déduisent :

$$
r_x(f, \text{Halfturn}, c) = r_x(f, \text{Clockwise}, r_x(f, \text{Clockwise}, c))
$$

$$
r_x(f, \text{Counterclockwise}, c) = r_x(f, \text{Clockwise}, r_x(f, \text{Clockwise}, r_x(f, \text{Clockwise}, c)))
$$


### Orientations

Pour toute face $f$ :

$$
r_\theta(f, \text{Halfturn}, c, o) = o
$$

- Rotation de la face droite :

$$
r_\theta(\text{Right}, d, c, o) =
\begin{cases}
    s_{0, 2}(o) & \text{si } c \in C_{\text{Right}} \\
    o & \text{sinon}
\end{cases}
$$

- Rotation de la face du bas :

$$
r_\theta(\text{Bottom}, d, c, o) =
\begin{cases}
    s_{0, 1}(o) & \text{si } c \in C_{\text{Bottom}} \\
    o & \text{sinon}
\end{cases}
$$

- Rotation de la face arrière :

$$
r_\theta(\text{Back}, d, c, o) =
\begin{cases}
    s_{1, 2}(o) & \text{si } c \in C_{\text{Back}} \\
    o & \text{sinon}
\end{cases}
$$


## Conditions d'arrêt

### Position finale

Le cube est correctement placé si :

$$
\forall id \in C : \quad x_{id, id}(t_{\text{max}})
$$

### Orientation finale

Et correctement orienté si :

$$
\forall c \in C : \quad \theta_{c, 0}(t_{\text{max}})
$$

## Transitions

Les transitions changent à la fois **la position** et **l'orientation** des cubes affectés.

Pour tout $t \in T^*$ :

### Position :

$$
a_{f, d}(t) \Rightarrow \left( x_{r_x(f,d,c), id}(t) = x_{c, id}(t-1) \right)
$$

Formulé en CNF :

- $(x_{c', id}(t) \lor \lnot x_{c, id}(t - 1) \lor \lnot a_{f, d}(t))$
- $(\lnot x_{c', id}(t) \lor x_{c, id}(t - 1) \lor \lnot a_{f, d}(t))$


### Orientation :

$$
a_{f, d}(t) \Rightarrow \left( \theta_{r_x(f,d,c), r_\theta(f,d,c,o)}(t) = \theta_{c, o}(t-1) \right)
$$

Formulé en CNF :

- $(\theta_{c', o'}(t) \lor \lnot \theta_{c, o}(t - 1) \lor \lnot a_{f,d}(t))$
- $(\lnot \theta_{c', o'}(t) \lor \theta_{c, o}(t - 1) \lor \lnot a_{f,d}(t))$


## Contraintes

### 1. Une seule action à la fois

$$
\forall t \in T^* :
\bigwedge_{\substack{(f, d), (f', d') \in F \times D \\ (f, d) <_{\text{lex}} (f', d')}} 
\left( \lnot a_{f, d}(t) \lor \lnot a_{f', d'}(t) \right)
$$

### 2. Toujours effectuer une action

$$
\forall t \in T^* :
\bigvee_{(f, d) \in F \times D} a_{f, d}(t)
$$
