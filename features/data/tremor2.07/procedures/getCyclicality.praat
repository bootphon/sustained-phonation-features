# # # # # # # # # # # # # # # T R E M O R # # # # # # # # # # # # # # # # #  
 #   g e t C y c l i c a l i t y . p r a a t   i s   a   P r a a t [ 6 . 0 . 0 6 ]   s c r i p t   ( h t t p : / / w w w . p r a a t . o r g / )    
 #   t h a t   s e r v e s   a s   a   p r o c e d u r e   w i t h i n   t r e m o r . p r a a t .  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   A u t h o r :   M a r k u s   B r � c k l   ( m a r k u s . b r u e c k l @ t u - b e r l i n . d e )  
 #   C o p y r i g h t   2 0 1 2 - 2 0 1 5   M a r k u s   B r � c k l  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   R e a d   " s t r e n g t h "   v a l u e s   f r o m   1   F R A M E   ( ! )   P r a a t   P i t c h   o j e c t s  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
 p r o c e d u r e   c y c l i  
  
 S a v e   a s   t e x t   f i l e :   " . / t e m p "  
 R e a d   S t r i n g s   f r o m   r a w   t e x t   f i l e :   " . / t e m p "  
  
 s t r i n g N   =   G e t   n u m b e r   o f   s t r i n g s  
 s N   =   0  
 f r e q   =   1 0 0  
 f o r   i s t r i n g   f r o m   1 0   t o   s t r i n g N  
       s e l e c t   S t r i n g s   t e m p  
       t E k s t $   =   G e t   s t r i n g :   i s t r i n g  
       t E k s t $   =   r e p l a c e $ ( t E k s t $ ,   "   " ,   " " , 1 0 0 )  
 # e c h o   ' i s t r i n g '   ' t E k s t $ '  
 # p a u s e  
       i f   s t a r t s W i t h ( t E k s t $ ,   " m a x n C a n d i d a t e s " )  
             c N   =   e x t r a c t N u m b e r ( t E k s t $ ,   " m a x n C a n d i d a t e s = " )  
             C r e a t e   s i m p l e   M a t r i x :   " t e m p " ,   ' c N ' ,   1 ,   " 0 "  
       e l s i f   s t a r t s W i t h ( t E k s t $ ,   " i n t e n s i t y " )  
             t r m   =   e x t r a c t N u m b e r ( t E k s t $ ,   " i n t e n s i t y = " )  
       e l s i f   s t a r t s W i t h ( t E k s t $ ,   " f r e q u e n c y " )  
             f r e q   =   e x t r a c t N u m b e r ( t E k s t $ ,   " f r e q u e n c y = " )  
       e l s i f   s t a r t s W i t h ( t E k s t $ ,   " s t r e n g t h " )   a n d   ( f r e q   < =   m a x T r )  
             s N   + = 1  
             s t r e n g t h   =   e x t r a c t N u m b e r ( t E k s t $ ,   " s t r e n g t h = " )  
             s e l e c t   M a t r i x   t e m p  
             S e t   v a l u e :   ' s N ' ,   1 ,   ' s t r e n g t h '  
 # e c h o   ' s N '   ' s t r e n g _ ' s N ' '  
 # p a u s e  
             e n d i f  
       e n d i f  
 e n d f o r  
 s e l e c t   M a t r i x   t e m p  
 t r c   =   G e t   m a x i m u m  
 # p a u s e  
 s e l e c t   S t r i n g s   t e m p  
 p l u s   M a t r i x   t e m p  
 R e m o v e  
 f i l e d e l e t e   . / t e m p  
  
 e n d p r o c
