 # # # # # # # # # # # # # # # T R E M O R # # # # # # # # # # # # # # # # #  
 #   a n a l y s i s i n o u t . p r a a t   i s   a   P r a a t [ 6 . 0 . 0 6 ]   s c r i p t   ( h t t p : / / w w w . p r a a t . o r g / )    
 #   t h a t   s e r v e s   a s   a   p r o c e d u r e   w i t h i n   t r e m o r . p r a a t .  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   A u t h o r :   M a r k u s   B r � c k l   ( m a r k u s . b r u e c k l @ t u - b e r l i n . d e )  
 #   C o p y r i g h t   2 0 1 2 - 2 0 1 5   M a r k u s   B r � c k l  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   S o u n d   ( . w a v )   i n ,   r e s u l t s   ( . t x t )   o u t  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 p r o c e d u r e   a n a i n o u t  
 p a u s e   R e c o r d / o p e n   a n d   s e l e c t   t h e   s o u n d   t o   b e   a n a l y z e d   ( i n   ' P r a a t   O b j e c t s ' ) !  
  
 i n f o $   =   I n f o  
 n a m e $   =   e x t r a c t W o r d $ ( i n f o $ ,   " O b j e c t   n a m e :   " )  
  
 s l e n g t h   =   G e t   t o t a l   d u r a t i o n  
  
 c a l l   f t r e m  
 c a l l   a t r e m  
  
 e c h o    
 . . . S o u n d n a m e :   ' n a m e $ ' ' n e w l i n e $ '  
 . . . ' n e w l i n e $ '  
 . . . f r e q u e n c y   c o n t o u r   m a g n i t u d e   ( F C o M ) :   ' f t r m : 3 ' ' n e w l i n e $ '  
 . . . a m p l i t u d e   c o n t o u r   m a g n i t u d e   ( A C o M ) :   ' a t r m : 3 ' ' n e w l i n e $ '  
 . . . ' n e w l i n e $ '  
 . . . f r e q u e n c y   t r e m o r   c y c l i c a l i t y   ( F T r C ) :   ' f t r c : 3 ' ' n e w l i n e $ '  
 . . . a m p l i t u d e   t r e m o r   c y c l i c a l i t y   ( A T r C ) :   ' a t r c : 3 ' ' n e w l i n e $ '  
 . . . ' n e w l i n e $ '  
 . . . f r e q u e n c y   t r e m o r   f r e q u e n c y   ( F T r F ) :   ' f t r f : 3 '   H z ' n e w l i n e $ '  
 . . . a m p l i t u d e   t r e m o r   f r e q u e n c y   ( A T r F ) :   ' a t r f : 3 '   H z ' n e w l i n e $ '  
 . . . ' n e w l i n e $ '  
 . . . f r e q u e n c y   t r e m o r   i n t e n s i t y   i n d e x   ( F T r I ) :   ' f t r i : 3 '   % ' n e w l i n e $ '  
 . . . a m p l i t u d e   t r e m o r   i n t e n s i t y   i n d e x   ( A T r I ) :   ' a t r i : 3 '   % ' n e w l i n e $ '  
 . . . ' n e w l i n e $ '  
 . . . f r e q u e n c y   t r e m o r   p o w e r   i n d e x   ( F T r P ) :   ' f t r p : 3 ' ' n e w l i n e $ '  
 . . . a m p l i t u d e   t r e m o r   p o w e r   i n d e x   ( A T r P ) :   ' a t r p : 3 ' ' n e w l i n e $ '  
  
 e n d p r o c
