# # # # # # # # # # # # # # # T R E M O R # # # # # # # # # # # # # # # # #  
 #   f r e q t r e m . p r a a t   i s   a   P r a a t [ 6 . 0 . 0 6 ]   s c r i p t   ( h t t p : / / w w w . p r a a t . o r g / )    
 #   t h a t   s e r v e s   a s   a   p r o c e d u r e   w i t h i n   t r e m o r . p r a a t .  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   A u t h o r :   M a r k u s   B r � c k l   ( m a r k u s . b r u e c k l @ t u - b e r l i n . d e )  
 #   C o p y r i g h t   2 0 1 2 - 2 0 1 5   M a r k u s   B r � c k l  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   F r e q u e n c y   T r e m o r   A n a l y s i s  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 p r o c e d u r e   f t r e m  
       T o   P i t c h   ( c c ) . . .   t s   m i n P i   1 5   y e s   s i l T h r e s h   v o i T h r e s h   o c C o   o c j C o   v u v C o   m a x P i  
  
 #i f   m o d e   =   1  
 #E d i t  
 #p a u s e   P i t c h   c o n t o u r  
 #e n d i f  
  
       n u m b e r V o i c e   =   C o u n t   v o i c e d   f r a m e s  
 i f   n u m b e r V o i c e   =   0  
       f t r c   =   u n d e f i n e d  
       f t r f   =   u n d e f i n e d  
       f t r i   =   u n d e f i n e d  
       f t r p   =   u n d e f i n e d  
 e l s e  
  
 #   b e c a u s e   P R A A T   o n l y   r u n s   " S u b t r a c t   l i n e a r   f i t "   i f   t h e   l a s t   f r a m e   i s   " v o i c e l e s s "   ( ! ? ) :  
 #   n u m b e r O f F r a m e s + 1   ( 1 )  
       n u m b e r O f F r a m e s   =   G e t   n u m b e r   o f   f r a m e s  
       x 1   =   G e t   t i m e   f r o m   f r a m e   n u m b e r . . .   1  
 #       a m _ F 0   =   G e t   m e a n . . .   0   0   H e r t z  
  
       C r e a t e   M a t r i x . . .   f t r e m _ 0   0   s l e n g t h   n u m b e r O f F r a m e s + 1   t s   x 1   1   1   1   1   1   0  
       f o r   i   f r o m   1   t o   n u m b e r O f F r a m e s  
             s e l e c t   P i t c h   ' n a m e $ '  
             f 0   =   G e t   v a l u e   i n   f r a m e . . .   i   H e r t z  
             s e l e c t   M a t r i x   f t r e m _ 0  
 #   w r i t e   z e r o s   t o   m a t r i x   w h e r e   f r a m e s   a r e   v o i c e l e s s  
             i f   f 0   =   u n d e f i n e d  
                   S e t   v a l u e . . .   1   i   0  
             e l s e  
                   S e t   v a l u e . . .   1   i   f 0  
             e n d i f  
       e n d f o r  
  
 #   r e m o v e   t h e   l i n e a r   F 0   t r e n d   ( F 0   d e c l i n a t i o n )  
       T o   P i t c h  
       S u b t r a c t   l i n e a r   f i t . . .   H e r t z  
       R e n a m e . . .   f t r e m _ 0 _ l i n  
  
 #   u n d o   ( 1 )  
       C r e a t e   M a t r i x . . .   f t r e m   0   s l e n g t h   n u m b e r O f F r a m e s   t s   x 1   1   1   1   1   1   0  
       f o r   i   f r o m   1   t o   n u m b e r O f F r a m e s  
             s e l e c t   P i t c h   f t r e m _ 0 _ l i n  
             f 0   =   G e t   v a l u e   i n   f r a m e . . .   i   H e r t z  
             s e l e c t   M a t r i x   f t r e m  
 #   w r i t e   z e r o s   t o   m a t r i x   w h e r e   f r a m e s   a r e   v o i c e l e s s  
             i f   f 0   =   u n d e f i n e d  
                   S e t   v a l u e . . .   1   i   0  
             e l s e  
                   S e t   v a l u e . . .   1   i   f 0  
             e n d i f  
       e n d f o r  
  
       T o   P i t c h  
       a m _ F 0   =   G e t   m e a n . . .   0   0   H e r t z  
  
 #   n o r m a l i z e   F 0 - c o n t o u r   b y   m e a n   F 0  
       s e l e c t   M a t r i x   f t r e m  
       F o r m u l a . . .   ( s e l f - a m _ F 0 ) / a m _ F 0  
  
 #   s i n c e   z e r o s   i n   t h e   M a t r i x   ( u n v o i c e d   f r a m e s )   b e c o m e   n o r m a l i z e d   t o   - 1   b u t    
 #   u n v o i c e d   f r a m e s   s h o u l d   b e   z e r o   ( i f   a n y t h i n g )  
 #   w r i t e   z e r o s   t o   m a t r i x   w h e r e   f r a m e s   a r e   v o i c e l e s s  
       f o r   i   f r o m   1   t o   n u m b e r O f F r a m e s  
             s e l e c t   P i t c h   f t r e m  
             f 0   =   G e t   v a l u e   i n   f r a m e . . .   i   H e r t z  
             i f   f 0   =   u n d e f i n e d  
                   s e l e c t   M a t r i x   f t r e m  
                   S e t   v a l u e . . .   1   i   0  
             e n d i f  
       e n d f o r  
  
 #   t o   c a l c u l a t e   a u t o c o r r e l a t i o n   ( c c - m e t h o d ) :  
       s e l e c t   M a t r i x   f t r e m  
       T o   S o u n d   ( s l i c e ) . . .   1  
 #   c a l c u l a t e   F r e q u e n c y   o f   F r e q u e n c y   T r e m o r   [ H z ]  
       T o   P i t c h   ( c c ) . . .   s l e n g t h   m i n T r   1 5   y e s   t r e m M a g T h r e s h   t r e m t h r e s h   o c F t r e m   0 . 3 5   0 . 1 4   m a x T r  
       R e n a m e . . .   f t r e m _ n o r m  
  
       f t r f   =   G e t   m e a n . . .   0   0   H e r t z  
  
       c a l l   c y c l i  
       f t r m   =   t r m  
       f t r c   =   t r c  
  
 #   c a l c u l a t e   I n t e n s i t y   I n d e x   o f   F r e q u e n c y   T r e m o r   [ % ]  
       s e l e c t   S o u n d   f t r e m  
       p l u s   P i t c h   f t r e m _ n o r m  
       T o   P o i n t P r o c e s s   ( p e a k s ) . . .   y e s   n o  
       R e n a m e . . .   M a x i m a  
       n u m b e r o f M a x P o i n t s   =   G e t   n u m b e r   o f   p o i n t s  
       f t r i _ m a x   =   0  
       n o F M a x   =   0  
       f o r   i P o i n t   f r o m   1   t o   n u m b e r o f M a x P o i n t s  
             s e l e c t   P o i n t P r o c e s s   M a x i m a  
             t i   =   G e t   t i m e   f r o m   i n d e x . . .   i P o i n t  
             s e l e c t   S o u n d   f t r e m  
             f t r i _ P o i n t   =   G e t   v a l u e   a t   t i m e . . .   A v e r a g e   t i   S i n c 7 0  
             i f   f t r i _ P o i n t   =   u n d e f i n e d  
                   f t r i _ P o i n t   =   0  
                   n o F M a x   + =   1  
             e n d i f  
             f t r i _ m a x   + =   a b s ( f t r i _ P o i n t )  
       e n d f o r  
  
 #i f   m o d e   =   1  
 #s e l e c t   S o u n d   f t r e m  
 #p l u s   P o i n t P r o c e s s   M a x i m a  
 #E d i t  
 #p a u s e   N o r m a l i z e d   a n d   d e - d e c l i n e d   p i t c h   c o n t o u r   a n d   m a x i m a  
 #e n d i f  
        
 #   f t r i _ m a x : =   ( m e a n )   p r o c e n t u a l   d e v i a t i o n   o f   F 0 - m a x i m a   f r o m   m e a n   F 0   a t   f t r f  
       n u m b e r o f M a x i m a   =   n u m b e r o f M a x P o i n t s   -   n o F M a x  
       f t r i _ m a x   =   1 0 0   *   f t r i _ m a x / n u m b e r o f M a x i m a  
  
       s e l e c t   S o u n d   f t r e m  
       p l u s   P i t c h   f t r e m _ n o r m  
       T o   P o i n t P r o c e s s   ( p e a k s ) . . .   n o   y e s  
       R e n a m e . . .   M i n i m a  
       n u m b e r o f M i n P o i n t s   =   G e t   n u m b e r   o f   p o i n t s  
       f t r i _ m i n   =   0  
       n o F M i n   =   0  
       f o r   i P o i n t   f r o m   1   t o   n u m b e r o f M i n P o i n t s  
             s e l e c t   P o i n t P r o c e s s   M i n i m a  
             t i   =   G e t   t i m e   f r o m   i n d e x . . .   i P o i n t  
             s e l e c t   S o u n d   f t r e m  
             f t r i _ P o i n t   =   G e t   v a l u e   a t   t i m e . . .   A v e r a g e   t i   S i n c 7 0  
             i f   f t r i _ P o i n t   =   u n d e f i n e d  
                   f t r i _ P o i n t   =   0  
                   n o F M i n   + =   1  
             e n d i f  
             f t r i _ m i n   + =   a b s ( f t r i _ P o i n t )  
       e n d f o r  
  
 #i f   m o d e   =   1  
 #s e l e c t   S o u n d   f t r e m  
 #p l u s   P o i n t P r o c e s s   M i n i m a  
 #E d i t  
 #p a u s e   N o r m a l i z e d   a n d   d e - d e c l i n e d   p i t c h   c o n t o u r   a n d   m i n i m a  
 #e n d i f  
  
 #   f t r i _ m i n : =   ( m e a n )   p r o c e n t u a l   d e v i a t i o n   o f   F 0 - m i n i m a   f r o m   m e a n   F 0   a t   f t r f  
       n u m b e r o f M i n i m a   =   n u m b e r o f M i n P o i n t s   -   n o F M i n  
       f t r i _ m i n   =   1 0 0   *   f t r i _ m i n / n u m b e r o f M i n i m a  
  
       f t r i   =   ( f t r i _ m a x   +   f t r i _ m i n )   /   2  
        
       f t r p   =   f t r i   *   f t r f / ( f t r f + 1 )  
  
 #   u n c o m m e n t   t o   i n s p e c t   f r e q u n e c y   t r e m o r   o b j e c t s :  
 #   p a u s e  
  
       s e l e c t   P i t c h   f t r e m  
 #   u n c o m m e n t   i f   o n l y   f r e q u e n c y   t r e m o r   i s   t o   b e   a n a l y z e d :  
 #       p l u s   P i t c h   ' n a m e $ '  
       p l u s   M a t r i x   f t r e m _ 0  
       p l u s   P i t c h   f t r e m _ 0  
       p l u s   P i t c h   f t r e m _ 0 _ l i n  
       p l u s   M a t r i x   f t r e m  
       p l u s   S o u n d   f t r e m  
       p l u s   P i t c h   f t r e m _ n o r m  
       p l u s   P o i n t P r o c e s s   M a x i m a  
       p l u s   P o i n t P r o c e s s   M i n i m a  
       R e m o v e  
  
 e n d i f  
 e n d p r o c
