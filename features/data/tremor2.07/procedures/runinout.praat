 # # # # # # # # # # # # # # # T R E M O R # # # # # # # # # # # # # # # # #  
 #   r u n i n o u t . p r a a t   i s   a   P r a a t [ 6 . 0 . 0 6 ]   s c r i p t   ( h t t p : / / w w w . p r a a t . o r g / )    
 #   t h a t   s e r v e s   a s   a   p r o c e d u r e   w i t h i n   t r e m o r . p r a a t .  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   A u t h o r :   M a r k u s   B r � c k l   ( m a r k u s . b r u e c k l @ t u - b e r l i n . d e )  
 #   C o p y r i g h t   2 0 1 2 - 2 0 1 5   M a r k u s   B r � c k l  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   S o u n d s   ( . w a v )   i n ,   r e s u l t s   ( . t x t )   o u t  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 p r o c e d u r e   r i n o u t  
 b e g i n P a u s e   ( " P a t h s " )  
       w o r d   ( " P a t h   o f   s o u n d s   t o   b e   a n a l y z e d " ,   " . . / " )  
       w o r d   ( " P a t h   a n d   n a m e   o f   r e s u l t   c s v " ,   " . . / r e s t a b _ t r e m o r " )  
 e n d P a u s e   ( " O K " ,   1 )  
  
 s o u r c e d i r e c $   =   p a t h _ o f _ s o u n d s _ t o _ b e _ a n a l y z e d $  
 r e s u l t d i r e c $   =   p a t h _ a n d _ n a m e _ o f _ r e s u l t _ c s v $  
  
 f i l e d e l e t e   ' r e s u l t d i r e c $ ' . c s v  
 f i l e a p p e n d   " ' r e s u l t d i r e c $ ' . c s v "    
 . . . s o u n d n a m e ' t a b $ '  
 . . . F C o M ' t a b $ '  
 . . . F T r C ' t a b $ '  
 . . . F T r F   [ H z ] ' t a b $ '  
 . . . F T r I   [ % ] ' t a b $ '  
 . . . F T r P ' t a b $ '  
 . . . A C o M ' t a b $ '  
 . . . A T r C ' t a b $ '  
 . . . A T r F   [ H z ] ' t a b $ '  
 . . . A T r I   [ % ] ' t a b $ '  
 . . . A T r P  
 . . . ' n e w l i n e $ '  
  
 C r e a t e   S t r i n g s   a s   f i l e   l i s t . . .   l i s t   ' s o u r c e d i r e c $ ' * . w a v  
 n u m b e r O f F i l e s   =   G e t   n u m b e r   o f   s t r i n g s  
 f o r   i f i l e   f r o m   1   t o   n u m b e r O f F i l e s  
       s e l e c t   S t r i n g s   l i s t  
       f i l e N a m e $   =   G e t   s t r i n g . . .   i f i l e  
       n a m e $   =   f i l e N a m e $   -   " . w a v "  
  
       R e a d   f r o m   f i l e . . .   ' s o u r c e d i r e c $ ' ' n a m e $ ' . w a v  
  
       s l e n g t h   =   G e t   t o t a l   d u r a t i o n  
  
       c a l l   f t r e m  
       c a l l   a t r e m  
  
       f i l e a p p e n d   " ' r e s u l t d i r e c $ ' . c s v "    
 . . . ' n a m e $ ' ' t a b $ '  
 . . . ' f t r m : 3 ' ' t a b $ '  
 . . . ' f t r c : 3 ' ' t a b $ '  
 . . . ' f t r f : 3 ' ' t a b $ '  
 . . . ' f t r i : 3 ' ' t a b $ '  
 . . . ' f t r p : 3 ' ' t a b $ '  
 . . . ' a t r m : 3 ' ' t a b $ '  
 . . . ' a t r c : 3 ' ' t a b $ '  
 . . . ' a t r f : 3 ' ' t a b $ '  
 . . . ' a t r i : 3 ' ' t a b $ '  
 . . . ' a t r p : 3 '  
 . . . ' n e w l i n e $ '  
  
       s e l e c t   S o u n d   ' n a m e $ '  
       R e m o v e  
 e n d f o r  
  
 s e l e c t   S t r i n g s   l i s t  
 R e m o v e  
  
 e n d p r o c
