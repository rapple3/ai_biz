# Sample policy documents for the airline chatbot

BAGGAGE_POLICY = """
# SkyWay Airlines Baggage Policy

## Carry-on Baggage
- All passengers are allowed one (1) carry-on bag and one (1) personal item.
- Maximum dimensions for carry-on: 22" x 14" x 9" (56 x 36 x 23 cm)
- Maximum dimensions for personal item: 18" x 14" x 8" (45 x 35 x 20 cm)
- Items must fit in overhead bin or under the seat in front of you

## Checked Baggage Allowance
- Standard passengers: First checked bag $30, Second checked bag $40
- Silver members: First checked bag free, Second checked bag $40
- Gold members: Two checked bags free
- Platinum members: Three checked bags free
- Maximum weight per bag: 50 lbs (23 kg)
- Maximum dimensions: 62 linear inches (158 cm) total

## Overweight/Oversized Baggage
- 51-70 lbs: $100 additional fee
- 71-100 lbs: $200 additional fee
- Oversized (63-80 linear inches): $100 additional fee

## Special Items
- Sports equipment: Special rules apply, see website for details
- Musical instruments: Can be carried on if they fit overhead or can be checked
- Mobility devices: Free of charge, do not count toward baggage allowance
"""

CANCELLATION_POLICY = """
# SkyWay Airlines Cancellation Policy

## Refundable Tickets
- Full refund if cancelled more than 24 hours before departure
- Cancellation fee of $200 applies if cancelled within 24 hours of departure
- No-shows will be charged the full ticket price

## Non-Refundable Tickets
- No refund available
- Value of ticket can be applied to future travel within 12 months, minus $200 change fee
- Changes must be made prior to scheduled departure

## Flight Disruptions
- If flight is cancelled by SkyWay Airlines: Full refund or rebooking on next available flight
- If flight is delayed more than 3 hours: Option to rebook or receive credit
- If flight is delayed more than 5 hours: Option for full refund

## Loyalty Member Benefits
- Platinum members: Change fees waived
- Gold members: Reduced change fee of $100
- Silver members: Reduced change fee of $150

## 24-Hour Flexible Booking Policy
- All tickets can be cancelled within 24 hours of booking for a full refund, provided the booking was made at least 7 days prior to departure
"""

REBOOKING_POLICY = """
# SkyWay Airlines Rebooking Policy

## Voluntary Changes
- Changes to non-refundable tickets: $200 change fee plus fare difference
- Changes to refundable tickets: No change fee, only fare difference applies
- Same-day flight change: $75 fee for standard passengers, free for Gold and Platinum members

## Involuntary Rebooking (Airline-Initiated)
- If flight is cancelled: Automatic rebooking on next available flight at no charge
- If rebooking is unsatisfactory: Option to choose alternative flight or receive refund
- Hotel accommodation provided for overnight delays due to airline operations

## Missed Connections
- If missed due to SkyWay Airlines delay: Automatic rebooking on next available flight
- If missed due to passenger delay: Standard change fees apply

## Loyalty Member Benefits
- Platinum members: Priority rebooking on full flights
- Gold members: Priority over standard passengers for rebooking
- All loyalty members: Access to dedicated rebooking hotline

## Name Changes
- Name changes are not permitted
- Tickets are non-transferable
"""

SPECIAL_ASSISTANCE = """
# SkyWay Airlines Special Assistance Policy

## Passengers with Disabilities
- Wheelchair assistance: Available free of charge, request at least 48 hours before departure
- Service animals: Permitted in cabin at no additional charge
- Emotional support animals: Require documentation submitted 48 hours before departure
- Accessible seating: Priority seating available for passengers with disabilities

## Unaccompanied Minors
- Service available for children ages 5-14
- Fee: $150 each way
- Must be booked at least 24 hours in advance
- Not available on connecting flights or last flight of the day

## Pregnant Passengers
- Medical certificate required for travel within 7 days of due date
- Not permitted to travel within 72 hours of due date
- No restrictions before 36 weeks

## Medical Conditions
- Passengers requiring medical oxygen must provide 48-hour notice
- CPAP machines permitted as additional carry-on item
- Passengers with recent surgeries may require medical clearance

## Allergy Concerns
- Nut-free buffer zones can be requested
- Special meals available with 24-hour advance notice
- Passengers with severe allergies should carry necessary medication
"""

LOYALTY_PROGRAM = """
# SkyWay Airlines Loyalty Program

## Membership Tiers
- Standard: Entry level, no minimum miles required
- Silver: 25,000 miles or 30 flight segments per calendar year
- Gold: 50,000 miles or 60 flight segments per calendar year
- Platinum: 100,000 miles or 100 flight segments per calendar year

## Miles Earning
- Economy class: 1 mile per mile flown
- Business class: 1.5 miles per mile flown
- First class: 2 miles per mile flown
- Partner airlines: Varies by partner and fare class

## Miles Redemption
- Domestic flights: Starting at 25,000 miles round trip
- International flights: Starting at 60,000 miles round trip
- Upgrades: Starting at 15,000 miles per segment
- Miles expire after 24 months of inactivity

## Tier Benefits
- Silver: Priority check-in, 1 free checked bag, 25% bonus miles
- Gold: Silver benefits plus priority boarding, 2 free checked bags, 50% bonus miles
- Platinum: Gold benefits plus lounge access, 3 free checked bags, 100% bonus miles, guaranteed availability

## Family Pooling
- Up to 8 family members can pool miles
- Primary account holder must be at least 18 years old
- All members earn tier status based on combined activity
"""

# Write the policies to files
with open('policies/baggage_policy.txt', 'w') as f:
    f.write(BAGGAGE_POLICY)

with open('policies/cancellation_policy.txt', 'w') as f:
    f.write(CANCELLATION_POLICY)

with open('policies/rebooking_policy.txt', 'w') as f:
    f.write(REBOOKING_POLICY)

with open('policies/special_assistance.txt', 'w') as f:
    f.write(SPECIAL_ASSISTANCE)

with open('policies/loyalty_program.txt', 'w') as f:
    f.write(LOYALTY_PROGRAM)

print("Policy documents created successfully!") 