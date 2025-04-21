import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {HomeRoutingModule} from './home-routing.module';
import {HomeComponent} from './Components/home/home.component';
import {StatisticsComponent} from './Components/home/Components/statistics/statistics.component';
import {AngularMaterialModule} from "../../../../Core/DesignModules/angular-material/angular-material.module";
import { ProfileComponent } from './Components/home/Components/profile/profile.component';
import { CandidatesComponent } from './Components/home/Components/candidates/candidates.component';
import { EntityProfileComponent } from './Components/home/Components/entity-profile/entity-profile.component';


@NgModule({
    declarations: [
        HomeComponent,
        StatisticsComponent,
        ProfileComponent,
        CandidatesComponent,
        EntityProfileComponent
    ],
    imports: [
        CommonModule,
        HomeRoutingModule,
        AngularMaterialModule
    ]
})
export class HomeModule {
}
