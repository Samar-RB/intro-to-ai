(define (problem instance_1)

	(:domain communityGarden)

	(:objects
		cell0 cell1 cell2 cell3 cell4 cell5 cell6 cell7 cell8 cell9 cell10 cell11 cell12 cell13 cell14 - location
		volunteer1 volunteer2 volunteer3 - volunteer
		tiller1 - tiller
		seeds1 - seeds
		wateringCan1 - wateringCan
	)

	(:init
		(at volunteer1 cell8)
		(at volunteer2 cell10)
		(at volunteer3 cell14)
		(at tiller1 cell4)
		(at seeds1 cell3)
		(at wateringCan1 cell7)
		(is-cultivator volunteer1)
		(is-planter volunteer2)
		(is-waterer volunteer3)
	)

	(:goal
		(and
			(garden-is-thriving)
		)
	)
)