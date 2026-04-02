(define (domain communityGarden)
     (:requirements :typing)
     (:types
          gardeningTool location volunteer - obj
          tiller seeds wateringCan - gardeningTool
     )

     (:constants
          gardenPlot - location
     )

     (:predicates
          (at ?o - obj ?loc - location)
          (has-tiller ?v - volunteer ?t - tiller)
          (has-seeds ?v - volunteer ?s - seeds)
          (has-watering-can ?v - volunteer ?wc - wateringCan)
          (is-cultivator ?v - volunteer)
          (is-planter ?v - volunteer)
          (is-waterer ?v - volunteer)
          (soil-is-tilled)
          (seeds-are-sown)
          (garden-is-watered)
          (garden-is-thriving)
     )

     ; how to start: 
     ; precondition - to change for each one below 
     ; effect - to change for each one below 
     ; at the end to check it with v1 v2 and to sumbit v3 v4 v5 
     ; stoped for today, tommorow to continue from till-soil.

     (:action move
          :parameters (?v - volunteer ?from - location ?to - location)
          :precondition (and 
               (at ?v ?from)
          )
          :effect (and
               (not (at ?v ?from))
               (at ?v ?to)
          )
     )

     (:action get-tiller
          :parameters (?v - volunteer ?t - tiller ?loc - location)
          :precondition (and
               (is-cultivator ?v)
               (at ?v ?loc)
               (at ?t ?loc)
          )
          :effect (and
               (has-tiller ?v ?t)
               (not (at ?t ?loc))
          )
     )

     (:action get-seeds
          :parameters (?v - volunteer ?s - seeds ?loc - location)
          :precondition (and
               (is-planter ?v)
               (at ?v ?loc)
               (at ?s ?loc)
          )
          :effect (and
               (has-seeds ?v ?s)
               (not (at ?s ?loc))
          )
     )


     (:action get-watering-can
          :parameters (?v - volunteer ?wc - wateringCan ?loc - location)
          :precondition (and
               (is-waterer ?v)
               (at ?v ?loc)
               (at ?wc ?loc)
          )
          :effect (and
               (has-watering-can ?v ?wc)
               (not (at ?wc ?loc))
          )
     )


     (:action till-soil
          :parameters (?v - volunteer ?tool - tiller)
          :precondition (and
               (is-cultivator ?v)
               (has-tiller ?v ?tool)
               (at ?v gardenPlot)
          )
          :effect (and
               (soil-is-tilled)
          )
     )


     (:action sow-seeds
          :parameters (?v - volunteer ?tool - seeds)
          :precondition (and
               (is-planter ?v)
               (has-seeds ?v ?tool)
               (at ?v gardenPlot)
               (soil-is-tilled)
          )
          :effect (and
               (seeds-are-sown)
          )
     )


     (:action water-garden
          :parameters (?v - volunteer ?tool - wateringCan)
          :precondition (and
               (is-waterer ?v)
               (has-watering-can ?v ?tool)
               (at ?v gardenPlot)
               (seeds-are-sown)
          )
          :effect (and
               (garden-is-watered)
          )
     )


     (:action celebrate-garden-opening
          :parameters (?c - volunteer ?p - volunteer ?w - volunteer)
          :precondition (and
               (is-cultivator ?c)
               (is-planter ?p)
               (is-waterer ?w)
               (at ?c gardenPlot)
               (at ?p gardenPlot)
               (at ?w gardenPlot)
               (soil-is-tilled)
               (seeds-are-sown)
               (garden-is-watered)
          )
          :effect (and
               (garden-is-thriving)
          )
     )

)