# # # # # # # # # # # # # # # T R E M O R # # # # # # # # # # # # # # # # #  
 #   a m p t r e m . p r a a t   i s   a   P r a a t [ 6 . 0 . 0 6 ]   s c r i p t   ( h t t p : / / w w w . p r a a t . o r g / )    
 #   t h a t   s e r v e s   a s   a   p r o c e d u r e   w i t h i n   t r e m o r . p r a a t .  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   A u t h o r :   M a r k u s   B r � c k l   ( m a r k u s . b r u e c k l @ t u - b e r l i n . d e )  
 #   C o p y r i g h t   2 0 1 2 - 2 0 1 5   M a r k u s   B r � c k l  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   A m p l i t u d e   T r e m o r   A n a l y s i s  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 p r o c e d u r e   a t r e m  
  
 s e l e c t   S o u n d   ' n a m e $ '  
 p l u s   P i t c h   ' n a m e $ '  
 T o   P o i n t P r o c e s s   ( c c )  
 n u m b O f G l o t P o i n t s   =   G e t   n u m b e r   o f   p o i n t s  
 i f   n u m b O f G l o t P o i n t s   <   3  
       a t r c   =   u n d e f i n e d  
       a t r f   =   u n d e f i n e d  
       a t r i   =   u n d e f i n e d  
       a t r p   =   u n d e f i n e d  
 e l s e  
       i f   a m p l i t u d e _ e x t r a c t i o n _ m e t h o d   =   2  
             s e l e c t   S o u n d   ' n a m e $ '  
             p l u s   P o i n t P r o c e s s   ' n a m e $ ' _ ' n a m e $ '  
 #   a m p l i t u d e s   a r e   R M S   p e r   p e r i o d   - -   n o t   i n t e n s i t y   m a x i m a   ? ?   - -   n o !   u n c l e a r ,   P r a a t   h e l p   m i s s i n g  
             T o   A m p l i t u d e T i e r   ( p e r i o d ) . . .   0   0   0 . 0 0 0 1   0 . 0 2   1 . 7  
             n u m b O f A m p P o i n t s   =   G e t   n u m b e r   o f   p o i n t s  
       e l s i f   a m p l i t u d e _ e x t r a c t i o n _ m e t h o d   =   1  
 #   N E W   2 0 1 4 - 1 2 - 1 3 :   c o r r e c t e d   m i s i n t e r p r e t a t i o n   o f   P r a a t - f u n c t i o n   " T o   A m p l i t u d e T i e r   ( p e r i o d ) "  
             n u m b O f A m p P o i n t s   =   n u m b O f G l o t P o i n t s   -   1  
             C r e a t e   A m p l i t u d e T i e r . . .   ' n a m e $ ' _ ' n a m e $ ' _ ' n a m e $ '   0   ' s l e n g t h '  
             f o r   i A m p P o i n t   f r o m   1   t o   n u m b O f A m p P o i n t s  
                   s e l e c t   P o i n t P r o c e s s   ' n a m e $ ' _ ' n a m e $ '  
                   p e r S t a r t   =   G e t   t i m e   f r o m   i n d e x . . .   i A m p P o i n t  
                   p e r E n d   =   G e t   t i m e   f r o m   i n d e x . . .   i A m p P o i n t + 1  
                   s e l e c t   S o u n d   ' n a m e $ '  
                   r m s   =   G e t   r o o t - m e a n - s q u a r e . . .   p e r S t a r t   p e r E n d  
 #   v e r y   s e l d o m l y   ( w i t h   b a d   p i t c h   s e t t i n g s )   i t   o c c u r s   t h a t   p e r S t a r t   a n d   p e r E n d   a r e   n e a r e r    
 #   t h a n   s a m p l i n g   p e r i o d   - >   r m s   w o u l d   b e   u n d e f i n e d  
                   i f   r m s   =   u n d e f i n e d  
                         s a m p l P e r   =   G e t   s a m p l i n g   p e r i o d  
                         r m s   =   G e t   r o o t - m e a n - s q u a r e . . .   p e r S t a r t - s a m p l P e r   p e r E n d + s a m p l P e r  
                   e n d i f  
                   s e l e c t   A m p l i t u d e T i e r   ' n a m e $ ' _ ' n a m e $ ' _ ' n a m e $ '  
                   A d d   p o i n t . . .   ( ' p e r S t a r t ' + ' p e r E n d ' ) / 2   r m s  
             e n d f o r  
       e n d i f  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
 #i f   m o d e   =   1  
 #E d i t  
 #p a u s e   A m p l i t u d e   c o n t o u r  
 #e n d i f  
  
 #   s i n c e   b a d   p i t c h   e x t r a c t i o n   m a y   r e s u l t   i n   n o t   e v e n   o n e   a m p l i t u d e   p o i n t  
 a m p P o i n t N   =   G e t   n u m b e r   o f   p o i n t s  
 i f   a m p P o i n t N   =   0  
       a t r c   =   u n d e f i n e d  
       a t r f   =   u n d e f i n e d  
       a t r i   =   u n d e f i n e d  
       a t r p   =   u n d e f i n e d  
 e l s e  
  
 #   f r o m   h e r e   o n   o u t :   p r e p a r e   t o   a u t o c o r r e l a t e   A m p l i t u d e T i e r - d a t a  
 #   s a m p l e   A m p l i t u d e T i e r   a t   ( c o n s t a n t )   r a t e   t s  
 #   t o   b e   a b l e   t o   - -   a u t o m a t i c a l l y   - -   r e a d   A m p .   v a l u e s . . .  
       D o w n   t o   T a b l e O f R e a l  
  
 #   t o   e n a b l e   a u t o c o r r e l a t i o n   o f   t h e   A m p . - c o n t o u r :   - > M a t r i x - > S o u n d  
  
       C r e a t e   M a t r i x . . .   a t r e m _ n l c   0   s l e n g t h   n u m b e r O f F r a m e s + 1   t s   x 1   1   1   1   1   1   0  
 #   f r o m   h e r e   o n   o u t :   g e t   t h e   m e a n   o f   ( t h e   c u r v e   o f )   t h e   a m p l i t u d e   c o n t o u r   i n   e a c h   f r a m e  
       f o r   i f r a m e   f r o m   1   t o   n u m b e r O f F r a m e s  
             s e l e c t   P i t c h   ' n a m e $ '  
             f 0   =   G e t   v a l u e   i n   f r a m e . . .   i f r a m e   H e r t z  
 #   d e t e r m i n e   ( t h e   t i m e   o f )   f i x e d   i n t e r v a l   b o r d e r s   f o r   t h e   r e s a m p l e d   a m p l i t u d e   c o n t o u r  
                   t   =   ( i f r a m e - 1 )   *   t s   +   x 1  
                   t l   =   t   -   t s / 2  
                   t u   =   t   +   t s / 2  
 #   g e t   t h e   i n d i c e s   o f   t h e   a m p l i t u d e   p o i n t s   s u r r o u n d i n g   a r o u n d   t h e s e   b o r d e r s  
                   s e l e c t   A m p l i t u d e T i e r   ' n a m e $ ' _ ' n a m e $ ' _ ' n a m e $ '  
                   l o i l   =   G e t   l o w   i n d e x   f r o m   t i m e . . .   t l  
                   h i i l   =   G e t   h i g h   i n d e x   f r o m   t i m e . . .   t l  
                   l o i u   =   G e t   l o w   i n d e x   f r o m   t i m e . . .   t u  
                   h i i u   =   G e t   h i g h   i n d e x   f r o m   t i m e . . .   t u  
 #   i f   t h e   s o u n d   i s   u n v o i c e d   t h e   a m p l i t u d e   i s   n o t   e x t r a c t e d  
             i f   f 0   =   u n d e f i n e d  
                   s e l e c t   M a t r i x   a t r e m _ n l c  
                   S e t   v a l u e . . .   1   i f r a m e   0  
 #   i f   t h e   a m p l i t u d e   c o n t o u r   h a s   n o t   b e g u n   y e t . . .  
             e l s i f   l o i l   =   0  
                   s e l e c t   M a t r i x   a t r e m _ n l c  
                   S e t   v a l u e . . .   1   i f r a m e   0  
 #   . . . o r   i s   a l r e a d y   f i n i s h e d   t h e   a m p l i t u d e   i s   n o t   e x t r a c t e d  
             e l s i f   h i i u   =   n u m b O f A m p P o i n t s   +   1 ;    
                   s e l e c t   M a t r i x   a t r e m _ n l c  
                   S e t   v a l u e . . .   1   i f r a m e   0  
             e l s e  
                   s e l e c t   T a b l e O f R e a l   ' n a m e $ ' _ ' n a m e $ ' _ ' n a m e $ '  
                   l o t l   =   G e t   v a l u e . . .   l o i l   1 ;   t i m e   v a l u e   o f   A m p . P o i n t   b e f o r e   t l   i n   t h e   P o i n t P r o c e s s   [ s ]  
                   d r u c k _ l o l   =   G e t   v a l u e . . .   l o i l   2 ;   a m p l i t u d e   v a l u e   b e f o r e   t l   i n   t h e   P o i n t P r o c e s s   [ P a ,   r a n g e d   f r o m   0   t o   1 ]  
                   h i t l   =   G e t   v a l u e . . .   h i i l   1  
                   d r u c k _ h i l   =   G e t   v a l u e . . .   h i i l   2 ;   a m p l i t u d e   v a l u e   a f t e r   t l   i n   t h e   P o i n t P r o c e s s  
                   l o t u   =   G e t   v a l u e . . .   l o i u   1  
                   d r u c k _ l o u   =   G e t   v a l u e . . .   l o i u   2 ;   a m p l i t u d e   v a l u e   b e f o r e   t u   i n   t h e   P o i n t P r o c e s s  
                   h i t u   =   G e t   v a l u e . . .   h i i u   1 ;   t i m e   v a l u e   a f t e r   t u   i n   t h e   P o i n t P r o c e s s  
                   d r u c k _ h i u   =   G e t   v a l u e . . .   h i i u   2 ;   a m p l i t u d e   v a l u e   a f t e r   t u   i n   t h e   P o i n t P r o c e s s  
 #   c a c u l a t e   ( l i n e a r l y   i n t e r p o l a t e d )   p r e s s u r e / a m p l i t u d e   a t   t h e   b o r d e r s  
                   d r u c k _ t l   =   ( ( h i t l - t l ) * d r u c k _ l o l   +   ( t l - l o t l ) * d r u c k _ h i l )   /   ( h i t l - l o t l )  
                   d r u c k _ t u   =   ( ( h i t u - t u ) * d r u c k _ l o u   +   ( t u - l o t u ) * d r u c k _ h i u )   /   ( h i t u - l o t u )  
  
                   n P i n t e r   =   h i i u   -   1   -   l o i l ;   =   l o i u   -   l o i l ;   =   h i i u   -   h i i l ;   n u m b e r   o f   a m p . - p o i n t s   b e t w e e n   t l   a n d   t u  
                   i f   n P i n t e r   =   0 ;   l o i l   =   l o i u ;   h i i l   =   h i i u  
                         d r u c k _ m e a n   =   ( d r u c k _ t l   +   d r u c k _ t u )   /   2  
                   e l s e  
                         t l i n t e r   =   t l  
                         p l i n t e r   =   d r u c k _ t l  
                         s u m t d r u c k   =   0  
                         f o r   i i n t e r   f r o m   1   t o   n P i n t e r  
                               t u i n t e r   =   G e t   v a l u e . . .   l o i l + i i n t e r   1  
                               p u i n t e r   =   G e t   v a l u e . . .   l o i l + i i n t e r   2  
                               d e l t a t   =   t u i n t e r   -   t l i n t e r  
                               t d r u c k _ i i n t e r   =   d e l t a t * ( p l i n t e r + p u i n t e r ) / 2  
                               s u m t d r u c k   + =   t d r u c k _ i i n t e r  
                               t l i n t e r   =   t u i n t e r  
                               p l i n t e r   =   p u i n t e r  
                         e n d f o r  
                         d e l t a t   =   t u   -   t l i n t e r  
                         t d r u c k _ i i n t e r   =   d e l t a t * ( p l i n t e r + d r u c k _ t u ) / 2  
                         s u m t d r u c k   + =   t d r u c k _ i i n t e r  
                         d r u c k _ m e a n   =   s u m t d r u c k   /   t s  
                   e n d i f  
  
                   s e l e c t   M a t r i x   a t r e m _ n l c  
                   S e t   v a l u e . . .   1   i f r a m e   d r u c k _ m e a n  
             e n d i f  
       e n d f o r  
  
 #   b e c a u s e   P R A A T   c l a s s i f i e s   f r e q u e n c i e s   i n   P i t c h   o b j e c t s   < = 0   a s   " v o i c e l e s s "   a n d    
 #   t h e r e f o r e   p a r t s   w i t h   e x t r e m e   I N T E N S I T I E S   w o u l d   b e   c o n s i d e r e d   a s   " v o i c e l e s s "  
 #   ( i r r e l e v a n t )   a f t e r   " S u b t r a c t   l i n e a r   f i t "   ( 1 )  
 #   " 1 "   i s   a d d e d   t o   t h e   o r i g i n a l   P a - v a l u e s   ( r a n g e d   f r o m   0   t o   1 )   - -   n o t   t o   t h e   v o i c e l e s s   p a r t s  
       s e l e c t   M a t r i x   a t r e m _ n l c  
       f o r   i   f r o m   1   t o   n u m b e r O f F r a m e s + 1  
             g r m s   =     G e t   v a l u e   i n   c e l l . . .   1   i  
             i f   g r m s   >   0  
                   S e t   v a l u e . . .   1   i   g r m s + 1  
             e n d i f  
       e n d f o r  
  
 #   r e m o v e   t h e   l i n e a r   a m p . - t r e n d   ( a m p l i t u d e   d e c l i n a t i o n )  
       T o   P i t c h  
       R e n a m e . . .   h i l f _ l i n c o r r  
  
       S u b t r a c t   l i n e a r   f i t . . .   H e r t z  
       R e n a m e . . .   a t r e m  
       a m _ I n t   =   G e t   m e a n . . .   0   0   H e r t z  
       a m _ I n t   =   a m _ I n t   -   1  
  
 #   u n d o   ( 1 ) . . .   a n d   n o r m a l i z e   A m p .   c o n t o u r   b y   m e a n   A m p .  
       T o   M a t r i x  
       f o r   i   f r o m   1   t o   n u m b e r O f F r a m e s + 1  
             g r m s   =     G e t   v a l u e   i n   c e l l . . .   1   i  
             i f   g r m s   >   0  
                   S e t   v a l u e . . .   1   i   ( g r m s - 1 - a m _ I n t ) / a m _ I n t  
             e n d i f  
       e n d f o r  
  
 #   r e m o v e   l a s t   f r a m e ,   u n d o   ( 2 )  
       C r e a t e   M a t r i x . . .   a t r e m _ b e s s e r   0   s l e n g t h   n u m b e r O f F r a m e s   t s   x 1   1   1   1   1   1   0  
       f o r   i f r a m e   f r o m   1   t o   n u m b e r O f F r a m e s  
             s e l e c t   M a t r i x   a t r e m  
             s p r i n g   =   G e t   v a l u e   i n   c e l l . . .   1   i f r a m e  
             s e l e c t   M a t r i x   a t r e m _ b e s s e r  
             S e t   v a l u e . . .   1   i f r a m e   s p r i n g  
       e n d f o r  
  
 #   t o   c a l c u l a t e   a u t o c o r r e l a t i o n   ( c c - m e t h o d )  
       T o   S o u n d   ( s l i c e ) . . .   1  
       T o   P i t c h   ( c c ) . . .   s l e n g t h   m i n T r   1 5   y e s   t r e m M a g T h r e s h   t r e m t h r e s h   o c A t r e m   0 . 3 5   0 . 1 4   m a x T r  
       R e n a m e . . .   a t r e m _ n o r m  
  
       a t r f   =   G e t   m e a n . . .   0   0   H e r t z  
  
       c a l l   c y c l i  
       a t r m   =   t r m  
       a t r c   =   t r c  
  
 #   c a l c u l a t e   I n t e n s i t y   I n d e x   o f   A m p l i t u d e   T r e m o r   [ % ]  
       s e l e c t   S o u n d   a t r e m _ b e s s e r  
       p l u s   P i t c h   a t r e m _ n o r m  
       T o   P o i n t P r o c e s s   ( p e a k s ) . . .   y e s   n o  
       R e n a m e . . .   M a x i m a  
       n u m b e r o f M a x P o i n t s   =   G e t   n u m b e r   o f   p o i n t s  
       a t r i _ m a x   =   0  
       n o A M a x   =   0  
       f o r   i P o i n t   f r o m   1   t o   n u m b e r o f M a x P o i n t s  
             s e l e c t   P o i n t P r o c e s s   M a x i m a  
             t i   =   G e t   t i m e   f r o m   i n d e x . . .   i P o i n t  
             s e l e c t   S o u n d   a t r e m _ b e s s e r  
             a t r i _ P o i n t   =   G e t   v a l u e   a t   t i m e . . .   0   t i   S i n c 7 0  
             i f   a t r i _ P o i n t   =   u n d e f i n e d  
                   a t r i _ P o i n t   =   0  
                   n o A M a x   + =   1  
             e n d i f  
             a t r i _ m a x   + =   a b s ( a t r i _ P o i n t )  
       e n d f o r  
  
 #i f   m o d e   =   1  
 #s e l e c t   S o u n d   a t r e m _ b e s s e r  
 #p l u s   P o i n t P r o c e s s   M a x i m a  
 #E d i t  
 #p a u s e   N o r m a l i z e d   a n d   d e - d e c l i n e d   a m p l i t u d e   c o n t o u r   a n d   m a x i m a  
 #e n d i f  
  
 #   a t r i _ m a x : =   ( m e a n )   p r o c e n t u a l   d e v i a t i o n   o f   A m p .   m a x i m a   f r o m   m e a n   A m p . [ P a ]   a t   a t r f  
       n u m b e r o f M a x i m a   =   n u m b e r o f M a x P o i n t s   -   n o A M a x  
       a t r i _ m a x   =   1 0 0   *   a t r i _ m a x   /   n u m b e r o f M a x i m a  
  
       s e l e c t   S o u n d   a t r e m _ b e s s e r  
       p l u s   P i t c h   a t r e m _ n o r m  
       T o   P o i n t P r o c e s s   ( p e a k s ) . . .   n o   y e s  
       R e n a m e . . .   M i n i m a  
       n u m b e r o f M i n P o i n t s   =   G e t   n u m b e r   o f   p o i n t s  
       a t r i _ m i n   =   0  
       n o A M i n   =   0  
       f o r   i P o i n t   f r o m   1   t o   n u m b e r o f M i n P o i n t s  
             s e l e c t   P o i n t P r o c e s s   M i n i m a  
             t i   =   G e t   t i m e   f r o m   i n d e x . . .   i P o i n t  
             s e l e c t   S o u n d   a t r e m _ b e s s e r  
             a t r i _ P o i n t   =   G e t   v a l u e   a t   t i m e . . .   0   t i   S i n c 7 0  
             i f   a t r i _ P o i n t   =   u n d e f i n e d  
                   a t r i _ P o i n t   =   0  
                   n o A M i n   + =   1  
             e n d i f  
             a t r i _ m i n   + =   a b s ( a t r i _ P o i n t )  
       e n d f o r  
  
 #if   m o d e   =   1  
 #s e l e c t   S o u n d   a t r e m _ b e s s e r  
 #p l u s   P o i n t P r o c e s s   M i n i m a  
 #E d i t  
 #p a u s e   N o r m a l i z e d   a n d   d e - d e c l i n e d   a m p l i t u d e   c o n t o u r   a n d   m i n i m a  
 #e n d i f  
  
 #   a t r i _ m i n : =   ( m e a n )   p r o c e n t u a l   d e v i a t i o n   o f   A m p .   m i n i m a   f r o m   m e a n   A m p . [ P a ]   a t   a t r f  
       n u m b e r o f M i n i m a   =   n u m b e r o f M i n P o i n t s   -   n o A M i n  
       a t r i _ m i n   =   1 0 0   *   a t r i _ m i n   /   n u m b e r o f M i n i m a  
  
       a t r i   =   ( a t r i _ m a x   +   a t r i _ m i n )   /   2  
  
       a t r p   =   a t r i   *   a t r f / ( a t r f + 1 )  
 e n d i f  
 e n d i f  
  
  
       s e l e c t   P i t c h   ' n a m e $ '  
       p l u s   P o i n t P r o c e s s   ' n a m e $ ' _ ' n a m e $ '  
 i f   n u m b O f G l o t P o i n t s   > =   3  
       p l u s   A m p l i t u d e T i e r   ' n a m e $ ' _ ' n a m e $ ' _ ' n a m e $ '  
 i f   a m p P o i n t N   >   0  
       p l u s   T a b l e O f R e a l   ' n a m e $ ' _ ' n a m e $ ' _ ' n a m e $ '  
       p l u s   M a t r i x   a t r e m _ n l c  
       p l u s   P i t c h   h i l f _ l i n c o r r  
       p l u s   P i t c h   a t r e m  
       p l u s   M a t r i x   a t r e m  
       p l u s   M a t r i x   a t r e m _ b e s s e r  
       p l u s   S o u n d   a t r e m _ b e s s e r  
       p l u s   P i t c h   a t r e m _ n o r m  
       p l u s   P o i n t P r o c e s s   M a x i m a  
       p l u s   P o i n t P r o c e s s   M i n i m a  
 e n d i f  
 e n d i f  
       R e m o v e  
  
 e n d p r o c
