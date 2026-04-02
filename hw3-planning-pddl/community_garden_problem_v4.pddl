(define (problem instance_4)

	(:domain communityGarden)

	(:objects
		cell0 cell1 cell2 cell3 cell4 cell5 cell6 cell7 cell8 cell9 cell10 cell11 cell12 cell13 cell14 cell15 cell16 cell17 cell18 cell19 cell20 cell21 cell22 cell23 cell24 cell25 cell26 cell27 cell28 cell29 cell30 cell31 cell32 cell33 cell34 - location
		volunteer1 volunteer2 volunteer3 - volunteer
		tiller1 - tiller
		seeds1 - seeds
		wateringCan1 - wateringCan
	)

	(:init
		(at volunteer1 cell28)
		(at volunteer2 cell32)
		(at volunteer3 cell16)
		(at tiller1 cell31)
		(at seeds1 cell15)
		(at wateringCan1 cell13)
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