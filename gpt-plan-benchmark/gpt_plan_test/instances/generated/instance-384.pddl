(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects h l j i e f k c)
(:init 
(handempty)
(ontable h)
(ontable l)
(ontable j)
(ontable i)
(ontable e)
(ontable f)
(ontable k)
(ontable c)
(clear h)
(clear l)
(clear j)
(clear i)
(clear e)
(clear f)
(clear k)
(clear c)
)
(:goal
(and
(on h l)
(on l j)
(on j i)
(on i e)
(on e f)
(on f k)
(on k c)
)))