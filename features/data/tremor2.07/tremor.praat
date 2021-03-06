# # # # # # # # # # # # # # # T R E M O R # # # # # # # # # # # # # # # # #  
 #   t r e m o r . p r a a t   i s   a   P r a a t [ 6 . 0 . 0 6 ]   s c r i p t   ( h t t p : / / w w w . p r a a t . o r g / )  
 #   t h a t   e x t r a c t s   1 0   m e a s u r e s   o f   v o c a l   t r e m o r   f r o m   a   w a v - f i l e   t h a t  
 #   c a p t u r e s   a   ( s u s t a i n e d )   p h o n a t i o n .  
 #   I n p u t :   w a v - f i l e  
 #   O u t p u t :  
 #   ( 1 )   f r e q u e n c y   c o n t o u r   m a g n i t u d e   ( F C o M ) ,  
 #   ( 2 )   f r e q u e n c y   t r e m o r   f r e q u e n c y   ( F T r F ) ,  
 #   ( 3 )   f r e q u e n c y   t r e m o r   c y c l i c a l i t y   ( F T r C ) ,  
 #   ( 4 )   f r e q u e n c y   t r e m o r   i n t e n s i t y   i n d e x   ( F T r I ) ,  
 #   ( 5 )   f r e q u e n c y   t r e m o r   p o w e r   i n d e x   ( F T r P ) ,  
 #   ( 6 )   a m p l i t u d e   c o n t o u r   m a g n i t u d e   ( A C o M ) ,  
 #   ( 7 )   a m p l i t u d e   t r e m o r   f r e q u e n c y   ( A T r F ) ,  
 #   ( 8 )   a m p l i t u d e   t r e m o r   c y c l i c a l i t y   ( A T r C ) ,  
 #   ( 9 )   a m p l i t u d e   t r e m o r   i n t e n s i t y   i n d e x   ( A T r I ) ,   a n d  
 #   ( 1 0 )   a m p l i t u d e   t r e m o r   p o w e r   i n d e x   ( A T r P ) .  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   D o c u m e n t a t i o n ,   h e l p :  
 #   B r � c k l ,   M .   ( 2 0 1 2 ) :   V o c a l   T r e m o r   M e a s u r e m e n t   B a s e d   o n   A u t o c o r r e l a t i o n    
 #   o f   C o n t o u r s .   I n :   P r o c e e d i n g s   o f   t h e   I S C A   C o n f e r e n c e   I n t e r s p e e c h   ' 1 2 ,    
 #   P o r t l a n d   ( O R ) .  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   A u t h o r :   M a r k u s   B r � c k l   ( m a r k u s . b r u e c k l @ t u - b e r l i n . d e )  
 #   C o p y r i g h t   2 0 1 2 - 2 0 1 5   M a r k u s   B r � c k l  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   1 . 0 0 :   t h e   v e r s i o n   u s e d   i n   B r � c k l   ( 2 0 1 1 )  
 #   2 . 0 1 :   t h e   v e r s i o n   p r e s e n t e d   i n   2 0 1 2   a t   I n t e r s p e e c h  
 #   n e w   i n . . .  
 #   2 . 0 2 :   m o d u l a r i z a t i o n   o f   t h e   s c r i p t   i n t o   s e p a r a t e   f i l e s  
 #   2 . 0 3 :   s e c o n d   w a y   t o   e x t r a c t   a n   A m p l i t u d e T i e r   f r o m   S o u n d   &   P o i n t P r o c e s s :   R M S   p e r    
 #               p i t c h   p e r i o d  
 #   2 . 0 4 :   r e m o v e d   a   b u g   i n   a m p t r e m . p r a a t   c a u s i n g   a   ( t h e   m o r e )   r a i s e d   " z e r o " - l e v e l   i n    
 #               a m p l i t u d e   c o n t o u r   ( t h e   l o w e r   t h e   m e a n   s o u n d   i n t e n s i t y   w a s )   a n d   t h e r e f o r e    
 #               r a i s e d   A T r I   a n d   A T r P   v a l u e s  
 #   2 . 0 5 :   r e m o v e d   a   b u g   c a u s i n g   t h e   s c r i p t   t o   s t o p   i f   t h e   s o u n d   i s   c o n s i d e r e d   v o i c e l e s s  
 #               a t   t h e   b e g i n n i g   o r   a t   t h e   e n d   - -   e . g .   b e c a u s e   o f   w r o n g   p i t c h   r a n g e  
 #   2 . 0 6 :   i m p r o v e d   t h e   r e s a m p l i n g   a t   a   c o n s t a n t   r a t e   ( t i m e   s t e p )   o f   t h e   a m p l i t u d e    
 #               c o n t o u r   ( p e r   p e r i o d )  
 #   2 . 0 7 :   o u t p u t   o f   m e a s u r e s   o f   t h e   c o n t o u r s '   ( m e a n )   m a g n i t u d e s   ( t e c h n i c a l l y   s p e a k i n g :    
 #               t h e   c o n t o u r s '   m e a n   ( r e l a t i v e )   d i s t a n c e   t o   z e r o   o r   a s   P r a a t   n a m e s   i t   i n   P i t c h    
 #               o b j e c t s :   ( a   f r a m e ' s )   i n t e n s i t y )  
 #   2 . 0 7 :   o u t p u t   o f   m e a s u r e s   f o r   t r e m o r   c y c l i c a l i t y   ( m e a s u r e s   o f   t r e m o r    
 #               p e r i o d i c i t y ,   t e c h n i c a l l y   s p e a k i n g :   t h e   c o n t o u r s '   a u t o - c o r r e l a t i o n   c o e f f i c i e n t s ,    
 #               o r   a s   P r a a t   n a m e s   i t   i n   P i t c h   o b j e c t s :   t h e   f r e q u e n c y ' s   " s t r e n g t h " )  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
  
  
  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 #   G l o b a l   S e t t i n g s  
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
 f o r m   T r e m o r   2 . 0 7  
 #       c o m m e n t   S e l e c t   p r o g r a m   m o d e  
       o p t i o n m e n u   M o d e   1  
             o p t i o n   A n a l y i s   m o d e  
             o p t i o n   R u n   m o d e  
       p o s i t i v e   A n a l y s i s _ t i m e _ s t e p _ ( s )   0 . 0 1 5  
 c o m m e n t   A r g u m e n t s   f o r   m a n d a t o r y   p i t c h   e x t r a c t i o n  
       p o s i t i v e   M i n i m a l _ p i t c h _ ( H z )   6 0  
       p o s i t i v e   M a x i m a l _ p i t c h _ ( H z )   3 5 0  
       p o s i t i v e   S i l e n c e _ t h r e s h o l d   0 . 0 3  
       p o s i t i v e   V o i c i n g _ t h r e s h o l d   0 . 3  
       p o s i t i v e   O c t a v e _ c o s t   0 . 0 1  
       p o s i t i v e   O c t a v e - j u m p _ c o s t   0 . 3 5  
       p o s i t i v e   V o i c e d _ / _ u n v o i c e d _ c o s t   0 . 1 4  
 c o m m e n t   A r g u m e n t s   f o r   t r e m o r   e x t r a c t i o n   f r o m   c o n t o u r s  
       o p t i o n m e n u   A m p l i t u d e _ e x t r a c t i o n _ m e t h o d   2  
             o p t i o n   I n t e g r a l   [ R M S   p e r   p i t c h   p e r i o d ]  
             o p t i o n   E n v e l o p e   [ T o   A m p l i t u d e T i e r   ( p e r i o d ) ]  
       p o s i t i v e   M i n i m a l _ t r e m o r _ f r e q u e n c y _ ( H z )   1 . 5  
       p o s i t i v e   M a x i m a l _ t r e m o r _ f r e q u e n c y _ ( H z )   1 5  
       p o s i t i v e   C o n t o u r _ m a g n i t u d e _ t h r e s h o l d   0 . 0 1  
       p o s i t i v e   T r e m o r _ c y c l i c a l i t y _ t h r e s h o l d   0 . 1 5  
       p o s i t i v e   F r e q u e n c y _ t r e m o r _ o c t a v e _ c o s t   0 . 0 1  
       p o s i t i v e   A m p l i t u d e _ t r e m o r _ o c t a v e _ c o s t   0 . 0 1  
 e n d f o r m  
  
 t s   =   a n a l y s i s _ t i m e _ s t e p ;   [ s ]  
  
 m i n P i   =   m i n i m a l _ p i t c h ;   [ H z ]  
 m a x P i   =   m a x i m a l _ p i t c h ;   [ H z ]  
 s i l T h r e s h   =   s i l e n c e _ t h r e s h o l d  
 v o i T h r e s h   =   v o i c i n g _ t h r e s h o l d  
 o c C o   =   o c t a v e _ c o s t  
 o c j C o   =   ' o c t a v e - j u m p _ c o s t '  
 v u v C o   =   ' v o i c e d _ / _ u n v o i c e d _ c o s t '  
  
 m i n T r   =   m i n i m a l _ t r e m o r _ f r e q u e n c y ;   [ H z ]  
 m a x T r   =   m a x i m a l _ t r e m o r _ f r e q u e n c y ;   [ H z ]  
 t r e m t h r e s h   =   t r e m o r _ c y c l i c a l i t y _ t h r e s h o l d  
 #   e q u a l s   " t h e   s t r e n g t h   o f   t h e   u n v o i c e d   c a n d i d a t e ,    
 #   r e l a t i v e   t o   t h e   m a x i m u m   p o s s i b l e   a u t o c o r r e l a t i o n . "   ( c f .   P r a a t   m a n u a l )  
 #   T h e   m a x .   p o s s i b l e   a u t o c o r r e l a t i o n   v a r i e s   ( e . g .   b e t w e e n   d i f f e r e n t   s o u n d s )  
 #   a n d   c o u l d   b e   a r o u n d   0 . 9 .  
 #   t r e m t h r e s h   =   t r e m t h r e s h   *   0 . 9  
 #   s a m e   f o r   t h e   3   o t h e r   t h r e s h o l d s  
 t r e m M a g T h r e s h   =   c o n t o u r _ m a g n i t u d e _ t h r e s h o l d  
  
 o c F t r e m   =   f r e q u e n c y _ t r e m o r _ o c t a v e _ c o s t  
 o c A t r e m   =   a m p l i t u d e _ t r e m o r _ o c t a v e _ c o s t  
  
  
 i n c l u d e   . / p r o c e d u r e s / a m p t r e m . p r a a t  
 i n c l u d e   . / p r o c e d u r e s / f r e q t r e m . p r a a t  
 i n c l u d e   . / p r o c e d u r e s / a n a l y s i s i n o u t . p r a a t  
 i n c l u d e   . / p r o c e d u r e s / r u n i n o u t . p r a a t  
 i n c l u d e   . / p r o c e d u r e s / g e t C y c l i c a l i t y . p r a a t  
  
 i f   m o d e   =   1  
       c a l l   a n a i n o u t  
 e l s e  
       c a l l   r i n o u t  
 e n d i f
